import torch
from torch import optim as optim

from iidp.profiler import IIDPCustomProfilerHelper, IIDPSingleGPUProfileTrainer

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
import models # [IIDP] This must be imported for t2t_vit models

from timm.loss import SoftTargetCrossEntropy
from timm.models import create_model


class ViTProfiler(IIDPCustomProfilerHelper):
    def __init__(self, lbs, num_models):
        super().__init__(lbs, num_models)

        self.model_name = 'vit'
        self.model = None
        self.criterion = SoftTargetCrossEntropy().to(self.gpu)
        self.optimizer = None
        self.param_groups_func = None

        self.weight_decay = 0.05
        self.opt_args = {'lr': 5e-4, 'weight_decay': self.weight_decay}

        self.prepare()

    def set_optimizer(self):
        def build_param_groups_in_optimizer(model):
            skip = {}
            if hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
                    no_decay.append(param)
                else:
                    decay.append(param)
            optimizer_grouped_parameters = [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': self.weight_decay}
            ]
            self.weight_decay = 0.
            return optimizer_grouped_parameters
        model = self.trainer.main_model
        self.param_groups_func = build_param_groups_in_optimizer
        optimizer_grouped_parameters = self.param_groups_func(model)
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, **self.opt_args)

    def prepare(self):
        torch.manual_seed(31415)
        self.trainer = IIDPSingleGPUProfileTrainer(
            self.gpu, self.lbs, self.num_models, self.accum_step, self.weight_sync_method)
        model_config = {
            'pretrained': False,
            'num_classes': 1000,
            'checkpoint_path': '',
            'drop_rate': 0.0,
            'drop_connect_rate': None,
            'drop_path_rate': 0.1,
            'drop_block': None,
            'gp': None,
            'bn_tf': False,
            'bn_momentum': None,
            'bn_eps': None,
            'img_size': 224
        }
        local_models = []
        for _ in range(self.trainer.num_local_models):
            model =  create_model('t2t_vit_14', **model_config).to(self.gpu)
            local_models.append(model)
        self.trainer.set_original_local_models(local_models)
        self.trainer.prepare_stream_parallel(local_models[0], self.criterion)
        self.set_optimizer()
        self.trainer.prepare_weight_sync_method(self.optimizer, None, self.param_groups_func)

    def run(self):
        print(f'====> Run IIDP profiler with the number of VSWs: {self.num_models}')
        self.trainer.set_model_train()
        input_shape = [self.trainer.batch_size_per_gpu, 3, 224, 224]
        for i in range(self.warmup_step+self.num_minibatches):
            dummy_images = torch.randn(*input_shape)
            # NOTE: target shape is different from resnet50 on ImageNet
            dummy_targets = torch.randint(1000, size=(self.trainer.batch_size_per_gpu, 1000),)
            with self.record_cuda_time():
                images = dummy_images.cuda(self.gpu, non_blocking=True)
                target = dummy_targets.cuda(self.gpu, non_blocking=True)

                scatter_images = torch.chunk(images, self.trainer.num_local_models)
                scatter_targets = torch.chunk(target, self.trainer.num_local_models)
            data_time = self.cuda_time
            fwd_time, bwd_time = self.trainer.profile_parallel_compute(scatter_images, scatter_targets)
            update_time, copy_time = self.trainer.profile_step()

            if i >= self.warmup_step:
                total_time = data_time + fwd_time + bwd_time + update_time + copy_time
                self.profile_data.update(data_time, fwd_time, bwd_time, update_time, copy_time, total_time)
                if i % 10 == 0:
                    print(f'[step {i}] {self.profile_data}')

        print(self.profile_data)
