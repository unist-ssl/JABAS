import torch
from torch.nn.utils import clip_grad_norm_

from iidp.profiler import IIDPCustomProfilerHelper, IIDPSingleGPUProfileTrainer

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing


class GNMTProfiler(IIDPCustomProfilerHelper):
    def __init__(self, lbs, num_models):
        super().__init__(lbs, num_models)

        self.model_name = 'gnmt'

        self.grad_clip = 5.0
        self.batch_first = False
        self.model_config = {
            'hidden_size': 1024,
            'vocab_size': 32317,
            'num_layers': 4,
            'dropout': 0, # For determenistic test, this value must be 0
            'batch_first': self.batch_first,
            'share_embedding': True,
        }
        self.model = GNMT(**self.model_config).to(self.gpu)
        pad_idx = 0
        smoothing = 0.1
        self.criterion = LabelSmoothing(pad_idx, smoothing, self.lbs).to(self.gpu)

        self.prepare()

    def prepare(self):
        torch.manual_seed(31415)
        self.trainer = IIDPSingleGPUProfileTrainer(
            self.gpu, self.lbs, self.num_models, self.accum_step, self.weight_sync_method)
        # To avoid RNN deepcopy issue
        local_models = []
        for _ in range(self.trainer.num_local_models):
            model =  GNMT(**self.model_config).to(self.gpu)
            local_models.append(model)
        self.trainer.set_original_local_models(local_models)
        self.trainer.prepare_stream_parallel(self.model, self.criterion)
        self.set_optimizer()
        self.trainer.prepare_weight_sync_method(self.optimizer, None, self.param_groups_func)

    def set_optimizer(self):
        lr = 2.00e-3
        self.opt_config = {'lr': lr}
        model = self.trainer.main_model
        self.optimizer = torch.optim.__dict__['Adam'](model.parameters(), **self.opt_config)

    def _prepare_local_batch_data(self):
        batch_size = self.trainer.local_batch_size
        if self.batch_first:
            src = torch.ones(batch_size, 50, dtype=torch.int64)
            src_length = torch.ones(batch_size, dtype=torch.int64)
            tgt = torch.ones(batch_size, 50, dtype=torch.int64)
        else:
            src = torch.ones(50, batch_size, dtype=torch.int64)
            src_length = torch.ones(batch_size, dtype=torch.int64)
            tgt = torch.ones(50, batch_size, dtype=torch.int64)

        src = src.to(self.gpu)
        src_length = src_length.to(self.gpu)
        tgt = tgt.to(self.gpu)
        if self.batch_first:
            parallel_input = [src, src_length, tgt[:, :-1]]
            parallel_target = tgt[:, 1:].contiguous().view(-1)
        else:
            parallel_input = [src, src_length, tgt[:-1]]
            parallel_target = tgt[1:].contiguous().view(-1)

        return parallel_input, parallel_target

    def run(self):
        print(f'====> Run IIDP profiler with the number of VSWs: {self.num_models}')
        self.trainer.set_model_train()
        for i in range(self.warmup_step+self.num_minibatches):
            with self.record_cuda_time():
                scatter_inputs, scatter_targets = [], []
                for _ in range(self.trainer.num_models):
                    parallel_input, parallel_target =  self._prepare_local_batch_data()
                    scatter_inputs.append(parallel_input)
                    scatter_targets.append(parallel_target)
            data_time = self.cuda_time
            fwd_time, bwd_time = self.trainer.profile_parallel_compute(scatter_inputs, scatter_targets)

            with self.record_cuda_time():
                clip_grad_norm_(self.trainer.main_model.parameters(), self.grad_clip)
            clip_time = self.cuda_time
            update_time, copy_time = self.trainer.profile_step()
            update_time += clip_time
            if i >= self.warmup_step:
                total_time = data_time + fwd_time + bwd_time + update_time + copy_time
                self.profile_data.update(data_time, fwd_time, bwd_time, update_time, copy_time, total_time)
                if i % 10 == 0:
                    print(f'[step {i}] {self.profile_data}')

        print(self.profile_data)
