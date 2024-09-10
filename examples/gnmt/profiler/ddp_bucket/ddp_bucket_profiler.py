import torch
from iidp.profiler import DDPHelper

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing


class GNMTProfiler(DDPHelper):
    def __init__(self):
        super().__init__()

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
        self.lbs = 32

    def _get_ddp_bucket_indices(self):
        self.ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
            self.model, device_ids=[self.gpu], output_device=[self.gpu],
            model_index=0, num_local_models=1, total_num_models=1)
        self.ddp_module.train()

        for _ in range(self.step):
            src = torch.ones(50, self.lbs, dtype=torch.int64).to(self.gpu)
            src_length = torch.ones(self.lbs, dtype=torch.int64).to(self.gpu)
            tgt = torch.ones(50, self.lbs, dtype=torch.int64).to(self.gpu)
            tgt_length = torch.randint(1, 49, (self.lbs,)).to(self.gpu)

            if self.batch_first:
                dummy_input = [src, src_length, tgt[:, :-1]]
                dummy_target = tgt[:, 1:].contiguous().view(-1)
            else:
                dummy_input = [src, src_length, tgt[:-1]]
                dummy_target = tgt[1:].contiguous().view(-1)
            output = self.ddp_module(*dummy_input)
            loss = self.criterion(output, dummy_target)
            loss.backward()

    def run(self):
        self.get_bucket_size_distribution()