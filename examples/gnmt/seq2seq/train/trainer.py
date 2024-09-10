# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import time

import torch
import torch.optim
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
from seq2seq.utils import AverageMeter


class Seq2SeqTrainer:
    """
    Seq2SeqTrainer
    """
    def __init__(self,
                 model,
                 batch_first,
                 grad_clip=float('inf'),
                 print_freq=10,
                 save_info={},
                 save_dir='.',
                 checkpoint_filename='checkpoint%s.pth',
                 translator=None,
                 verbose=False,
                 num_minibatches=None,
                 dp_trainer=None,
                 is_adaptive_training=False):
        """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_info: dict with additional state stored in each checkpoint
        :param save_dir: path to the directiory for checkpoints
        :param checkpoint_filename: name of files with checkpoints
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.epoch = 0
        self.device = next(model.parameters()).device
        self.print_freq = print_freq
        self.verbose = verbose
        self.loss = None
        self.translator = translator

        # checkpoint configuration
        self.save_info = save_info
        self.save_dir = save_dir
        self.checkpoint_filename = checkpoint_filename

        self.distributed = torch.distributed.is_initialized()
        self.batch_first = batch_first

        self.grad_clip = grad_clip

        self.num_minibatches = num_minibatches
        self.dp_trainer = dp_trainer

        self.optimizer = dp_trainer.main_optimizer
        self.scheduler = dp_trainer.main_scheduler
        self.criterion = dp_trainer.criterion

        self.is_adaptive_training = is_adaptive_training

    def _generate_scatter_inputs_and_targets(self, src, tgt):
        def _scatter_sentence(sentence, chunk_num):
            tgt1, tgt2 = sentence
            tgt1s = torch.chunk(tgt1, chunk_num, dim=1)
            tgt2s = torch.chunk(tgt2, chunk_num, dim=0)
            result = list(zip(tgt1s, tgt2s))
            return result
        scatter_inputs, scatter_targets = [], []
        num_models = self.dp_trainer.num_local_models

        scatter_srcs = _scatter_sentence(src, num_models)
        scatter_tgts = _scatter_sentence(tgt, num_models)

        for scatter_src, scatter_tgt in zip(scatter_srcs, scatter_tgts):
            scatter_src, src_length = scatter_src
            scatter_tgt, _ = scatter_tgt

            scatter_src = scatter_src.to(self.device)
            src_length = src_length.to(self.device)
            scatter_tgt = scatter_tgt.to(self.device)

            if self.batch_first:
                scatter_input = [scatter_src, src_length, scatter_tgt[:, :-1]]
                scatter_target = scatter_tgt[:, 1:].contiguous().view(-1)
            else:
                scatter_input = [scatter_src, src_length, scatter_tgt[:-1]]
                scatter_target = scatter_tgt[1:].contiguous().view(-1)
            scatter_inputs.append(scatter_input)
            scatter_targets.append(scatter_target)

        return scatter_inputs, scatter_targets

    def train(self, data_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        comp_time = AverageMeter()
        update_time = AverageMeter()

        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        # token metric
        num_toks = {}
        tot_tok_perf = AverageMeter()
        src_tok_perf = AverageMeter()
        tgt_tok_perf = AverageMeter()

        end = time.time()
        warmup_step = 10 * (self.dp_trainer.accum_step+1)
        for i, (data, num_toks) in enumerate(data_loader):
            if self.num_minibatches is not None and self.dp_trainer.sync_step > self.num_minibatches:
                break
            is_record_step = ((i+1) % (self.dp_trainer.accum_step+1) == 0)
            # measure data loading time
            if i >= warmup_step and is_record_step:
                data_time.update(time.time() - end)
                start = time.time()

            self.dp_trainer.compute(data)
            # record loss - NOTE: Since loss is divided by local batch size, we multiply it
            # Loss - LabelSmoothing in [seq2seq/train/smoothing.py]
            loss_per_batch = self.dp_trainer.losses[0].detach() * self.dp_trainer.local_batch_size

            if i >= warmup_step and is_record_step:
                comp_time.update(time.time() - start)
                start = time.time()

            if (not self.dp_trainer.weight_sync_method == 'overlap' and \
                    self.grad_clip != float('inf') and \
                    self.dp_trainer.is_sync_step()):
                with torch.cuda.stream(self.dp_trainer.main_stream):
                    clip_grad_norm_(self.dp_trainer.main_model.parameters(), self.grad_clip)

            is_sync_step = self.dp_trainer.step()
            if is_sync_step:
                self.dp_trainer.scheduler_step()

            if i >= warmup_step and is_record_step:
                update_time.update(time.time() - start)

            # measure elapsed time
            if is_record_step:
                elapsed = time.time() - end
                end = time.time()
            if i >= warmup_step and is_record_step:
                batch_time.update(elapsed)

                src_tok_perf.update(num_toks['src'] / elapsed)
                tgt_tok_perf.update(num_toks['tgt'] / elapsed)
                tot_num_toks = num_toks['tgt'] + num_toks['src']
                tot_tok_perf.update(tot_num_toks / elapsed)

            # record loss
            loss_per_token = loss_per_batch / num_toks['tgt']
            losses_per_token.update(loss_per_token, num_toks['tgt'])

            if self.is_adaptive_training:
                is_logging_step = (is_record_step and (self.dp_trainer.sync_step % self.print_freq == 0))
            else:
                is_logging_step = (is_record_step and ((data_loader.step_index+1) % self.print_freq == 0))
            if is_logging_step:
                log = []
                if self.is_adaptive_training:
                    log += [f'[{self.epoch}][{data_loader.data_index}/{len(data_loader.dataset)}]']
                else:
                    if self.num_minibatches is not None:
                        log += [f'[{self.epoch}][{data_loader.step_index+1}/{self.num_minibatches}]']
                    else:
                        log += [f'[{self.epoch}][{data_loader.step_index+1}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.4f} ({data_time.avg:.5f})']
                log += [f'Comp {comp_time.val:.5f} ({comp_time.avg:.5f})']
                log += [f'Update {update_time.val:.5f} ({update_time.avg:.5f})']
                log += [f'Tok/s {tot_tok_perf.val:.0f} ({tot_tok_perf.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_perf.val:.0f} ({src_tok_perf.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_perf.val:.0f} ({tgt_tok_perf.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                lr = self.optimizer.param_groups[0]['lr']
                log += [f'LR {lr:.3e}']
                log = '\t'.join(log)
                if torch.distributed.get_rank() == 0:
                    logging.info(log)

        tot_tok_perf.reduce('sum')
        losses_per_token.reduce('mean')

        return losses_per_token.avg, tot_tok_perf.avg

    def validate(self, data_loader):
        batch_time = AverageMeter()

        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size

        end = time.time()
        for i, (src, tgt) in enumerate(data_loader):
            src, src_length = src
            tgt, tgt_length = tgt
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_length = src_length.to(self.device)

            num_toks = {}
            num_toks['tgt'] = int(sum(tgt_length - 1))
            num_toks['src'] = int(sum(src_length))
            if self.batch_first:
                output = self.model(src, src_length, tgt[:, :-1])
                tgt_labels = tgt[:, 1:]
            else:
                output = self.model(src, src_length, tgt[:-1])
                tgt_labels = tgt[1:]
            target = tgt_labels.contiguous().view(-1)
            loss = self.criterion(output, target)
            # record loss - NOTE: Since loss is divided by local batch size, we multiply it
            # Loss - LabelSmoothing in [seq2seq/train/smoothing.py]
            loss_per_batch = loss.item() * batch_size
            loss_per_token = loss_per_batch / num_toks['tgt']
            loss_per_sentence = loss_per_batch / batch_size

            # measure elapsed time
            elapsed = time.time() - end
            end = time.time()

            # record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            if i >= 10:
                batch_time.update(elapsed)
                src_tok_time.update(num_toks['src'] / elapsed)
                tgt_tok_time.update(num_toks['tgt'] / elapsed)
                tot_num_toks = num_toks['tgt'] + num_toks['src']
                tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg

            if i % self.print_freq == 0:
                log = []
                log += [f'[VALIDATION][{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.4f} ({losses_per_token.avg:.4f})']
                log = '\t'.join(log)
                logging.info(log)

        tot_tok_time.reduce('sum')
        losses_per_token.reduce('mean')

        return losses_per_token.avg, tot_tok_time.avg

    def preallocate(self, batch_size, max_length):
        logging.info('Executing preallocation')
        torch.cuda.empty_cache()
        # Prepare the dummy input with maximum sequence length
        src_length = torch.full((batch_size,), max_length,
                                dtype=torch.int64)
        tgt_length = torch.full((batch_size,), max_length,
                                dtype=torch.int64)

        if self.batch_first:
            shape = (batch_size, max_length)
        else:
            shape = (max_length, batch_size)

        src = torch.full(shape, 4, dtype=torch.int64)
        tgt = torch.full(shape, 4, dtype=torch.int64)
        src = src, src_length
        tgt = tgt, tgt_length

        # Run forward and backward in training without updating parameters
        scatter_inputs, scatter_targets = self._generate_scatter_inputs_and_targets(src, tgt)
        self.dp_trainer.parallel_compute(scatter_inputs, scatter_targets, accum_step=0)
        for local_model in self.dp_trainer.local_models:
            if self.grad_clip != float('inf'):
                clip_grad_norm_(local_model.parameters(), self.grad_clip)

        # Prevent from accumulating gradient from dummy input
        for local_model in self.dp_trainer.local_models:
            local_model.zero_grad()

    def optimize(self, data_loader):
        """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(True)
        self.dp_trainer.set_model_train()
        if self.is_adaptive_training is False:
            if self.dp_trainer.is_accum_mode:
                logging.info('[INFO] No pre-allocate if accum mode')
            else:
                self.preallocate(data_loader.batch_size, data_loader.dataset.max_len)

        output = self.train(data_loader)

        return output

    def evaluate(self, data_loader):
        """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        output = self.validate(data_loader)
        return output

    def load(self, filename):
        """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            for local_model in self.dp_trainer.local_models:
                if self.dp_trainer.prepared_for_ddp:
                    local_model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    local_model.load_state_dict(checkpoint['state_dict'])
            for local_optimizer in self.dp_trainer.local_optimizers:
                local_optimizer.load_state_dict(checkpoint['optimizer'])
            for local_scheduler in self.dp_trainer.local_schedulers:
                local_scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'Invalid checkpoint: {filename}')

    def save(self, is_best=False, is_save=False):
        """
        Stores checkpoint to a file.

        :param is_best: if True stores checkpoint to 'model_best.pth'
        :param is_save: if True stores checkpoint after completed training
            epoch
        """

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_dir, filename)
            logging.info(f'Saving model to {filename}')
            torch.save(state, filename)

        if not self.save_dir:
            print('[INFO] No directory to save ckpt')
            return

        model_state = self.model.state_dict()
        scheduler_state = self.scheduler.state_dict()
        if self.is_adaptive_training:
            scheduler_state.pop('data_loader')
        state = {
            'epoch': self.epoch+1,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_state,
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if is_save:
            filename = f'checkpoint.pth.tar'
            write_checkpoint(state, filename)
