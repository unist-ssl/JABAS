# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import utils
import dllogger

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(
        self,
        arch,
        loss,
        pretrained_weights=None,
        cuda=True,
        memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1], arch[2])
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        # if cuda:
            # model = model.cuda().to(memory_format=memory_format)

        # define loss function (criterion) and optimizer
        criterion = loss()

        # if cuda:
            # criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    # def distributed(self, gpu_id):
        # self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(
    parameters,
    lr,
    momentum,
    weight_decay,
    nesterov=False,
    state=None,
    bn_weight_decay=False,
):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD(
            [v for n, v in parameters],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ],
            lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None
):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier) / es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)


# def get_train_step(
#     model_and_loss, optimizer, scaler, use_amp=False, batch_size_multiplier=1
# ):
#     def _step(input, target, optimizer_step=True):
#         input_var = Variable(input)
#         target_var = Variable(target)

#         # with autocast(enabled=use_amp):
#         loss, output = model_and_loss(input_var, target_var)
#         loss /= batch_size_multiplier

#         model_and_loss.model.backward()

#         if optimizer_step:

#             optimizer.zero_grad()

#         torch.cuda.synchronize()

#         return reduced_loss

#     return _step


def train(
    train_loader,
    model_and_loss,
    optimizer,
    scaler,
    lr_scheduler,
    logger,
    epoch,
    timeout_handler,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
    summary_writer=None,
):


    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    # step = get_train_step(
    #     model_and_loss,
    #     optimizer,
    #     scaler=scaler,
    #     use_amp=use_amp,
    #     batch_size_multiplier=batch_size_multiplier,
    # )

    model_and_loss.model.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    global global_update_step_count
    global global_data_sample_count
    inputs=[]
    targets=[]
    for i, (single_input, single_target) in data_iter:
        model=model_and_loss.model
        #qy: flexible micro batch size
        inputs.append(single_input)
        targets.append(single_target)
        if len(inputs)<model.train_micro_batch_size_per_gpu():
            continue
        else:
            input=torch.cat(inputs)
            target=torch.cat(targets)
            inputs=[]
            targets=[]
            
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0

        # loss = step(input, target, optimizer_step=optimizer_step)
        input, target = input.to(model.local_rank), target.to(model.local_rank)
        output = model(input)
        loss = model_and_loss.loss(output, target)
        model.backward(loss)
        global_data_sample_count+=bs*torch.distributed.get_world_size()
        if model.is_gradient_accumulation_boundary():
            global_update_step_count+=1
            for param_group in optimizer.param_groups:
                lr_this_step = param_group['lr']
            _epoch = global_data_sample_count
            if (torch.distributed.get_rank() == 0) or (torch.distributed.get_rank() == -1):
                # dist_print(_epoch)
                summary_writer.add_scalar(f'cos_similarity', model.cos_placeholder if model.cos_placeholder else float('NaN'),
                                               _epoch)
                summary_writer.add_scalar(f'adjust_step', model.gradient_step_size if model.gradient_step_size else float('NaN'),
                                               _epoch)
                summary_writer.add_scalar(f'accumulation_steps', model.gradient_accumulation_steps(),
                                               _epoch)
                summary_writer.add_scalar(f'number_of_samples_vs_update_steps', _epoch,
                                               global_update_step_count)
                try:
                    summary_writer.add_scalar(
                        f'learning_rate', lr_this_step*model.adaptive_batch_params["global_lr_modifier"], _epoch)
                except:
                    summary_writer.add_scalar(
                        f'learning_rate', lr_this_step, _epoch)

                summary_writer.add_scalar(f'adjusted_batch_size', model.gradient_accumulation_steps(
                )*torch.distributed.get_world_size()*model.train_micro_batch_size_per_gpu(), _epoch)
        model.step()

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric("train.loss", loss.item(), bs)
            logger.log_metric("train.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()
        if timeout_handler and timeout_handler.interrupted:
            break


def get_val_step(model_and_loss, use_amp=False):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad(), autocast(enabled=use_amp):
            loss, output = model_and_loss(input_var, target_var)

            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data)
                prec1 = utils.reduce_tensor(prec1)
                prec5 = utils.reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    val_loader,
    model_and_loss,
    logger,
    epoch,
    use_amp=False,
    prof=-1,
    register_metrics=True,
    summary_writer=None,
):
    if register_metrics and logger is not None:
        logger.register_metric(
            "val.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            "val.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "val.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "val.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "val.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss, use_amp=use_amp)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.model.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)
    if prof > 0:
        data_iter = utils.first_n(prof, data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(prec1.item(), bs)
        if logger is not None:
            logger.log_metric("val.top1", prec1.item(), bs)
            logger.log_metric("val.top5", prec5.item(), bs)
            logger.log_metric("val.loss", loss.item(), bs)
            logger.log_metric("val.compute_ips", calc_ips(bs, it_time - data_time))
            logger.log_metric("val.total_ips", calc_ips(bs, it_time))
            logger.log_metric("val.data_time", data_time)
            logger.log_metric("val.compute_latency", it_time - data_time)
            logger.log_metric("val.compute_latency_at95", it_time - data_time)
            logger.log_metric("val.compute_latency_at99", it_time - data_time)
            logger.log_metric("val.compute_latency_at100", it_time - data_time)


        end = time.time()
    print(top1.get_val())
    if torch.distributed.get_rank()<1:
        summary_writer.add_scalar("val_top1_acc",top1.get_val()[0],epoch)

    return top1.get_val()


# Train loop {{{
def calc_ips(batch_size, time):
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    tbs = world_size * batch_size
    return tbs / time


def train_loop(
    model_and_loss,
    optimizer,
    scaler,
    lr_scheduler,
    train_loader,
    val_loader,
    logger,
    should_backup_checkpoint,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    summary_writer=None,
):
    prec1 = -1

    global global_update_step_count
    global global_data_sample_count
    global_update_step_count=0
    global_data_sample_count=0
    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    #with utils.TimeoutHandler() as timeout_handler:
    timeout_handler = None
    for epoch in range(start_epoch, end_epoch):
        if logger is not None:
            logger.start_epoch()
        if not skip_training:
            train(
                train_loader,
                model_and_loss,
                optimizer,
                scaler,
                lr_scheduler,
                logger,
                epoch,
                timeout_handler,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                batch_size_multiplier=batch_size_multiplier,
                summary_writer=summary_writer,
            )

        if not skip_validation:
            prec1, nimg = validate(
                val_loader,
                model_and_loss,
                logger,
                epoch,
                use_amp=use_amp,
                prof=prof,
                register_metrics=epoch == start_epoch,
                summary_writer=summary_writer,
            )
        if logger is not None:
            logger.end_epoch()

        # if save_checkpoints and (
        #     not torch.distributed.is_initialized()
        #     or torch.distributed.get_rank() == 0
        # ):
            # if not skip_validation:
            #     is_best = (
            #         logger.metrics["val.top1"]["meter"].get_epoch() > best_prec1
            #     )
            #     best_prec1 = max(
            #         logger.metrics["val.top1"]["meter"].get_epoch(), best_prec1
            #     )
            # else:
            #     is_best = False
            #     best_prec1 = 0

            # if should_backup_checkpoint(epoch):
            #     backup_filename = "checkpoint-{}.pth.tar".format(epoch + 1)
            # else:
            #     backup_filename = None
            # utils.save_checkpoint(
            #     {
            #         "epoch": epoch + 1,
            #         "arch": model_and_loss.arch,
            #         "state_dict": model_and_loss.model.state_dict(),
            #         "best_prec1": best_prec1,
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     is_best,
            #     checkpoint_dir=checkpoint_dir,
            #     backup_filename=backup_filename,
            #     filename=checkpoint_filename,
            # )
            #if timeout_handler.interrupted:
            #    break



# }}}
