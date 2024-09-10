import torch

from iidp.profiler import IIDPCustomProfilerHelper

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
import image_classification.resnet as models
from image_classification.smoothing import LabelSmoothing


class ResNet50Profiler(IIDPCustomProfilerHelper):
    def __init__(self, lbs, num_models):
        super().__init__(lbs, num_models)

        self.model_name = 'resnet50'
        self.model_config = 'fanin'
        self.num_classes = 1000
        self.label_smoothing = 0.1
        self.model = models.build_resnet(self.model_name, self.model_config, self.num_classes).to(self.gpu)
        loss = lambda: LabelSmoothing(self.label_smoothing)
        self.criterion = loss().to(self.gpu)

        self.lr = 2.048
        self.momentum = 0.875
        self.weight_decay = 3.0517578125e-05

        self.prepare()

        self.auxiliary_profile_data = {'bn_sync_time': 6.29}

    def set_optimizer(self):
        def build_param_groups_in_optimizer(model):
            parameters = list(model.named_parameters())
            bn_params = [v for n, v in parameters if "bn" in n]
            rest_params = [v for n, v in parameters if not "bn" in n]
            optimizer_grouped_parameters  = [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": self.weight_decay},
            ]
            return optimizer_grouped_parameters
        model = self.trainer.main_model
        params = build_param_groups_in_optimizer(model)
        self.optimizer = torch.optim.SGD(params, self.lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.param_groups_func = build_param_groups_in_optimizer

    def run(self):
        print(f'====> Run IIDP profiler with the number of VSWs: {self.num_models}')
        self.trainer.set_model_train()
        input_shape = [self.trainer.batch_size_per_gpu, 3, 224, 224]
        for i in range(self.warmup_step+self.num_minibatches):
            dummy_images = torch.randn(*input_shape)
            dummy_targets = torch.empty(self.trainer.batch_size_per_gpu, dtype=torch.int64).random_(1000)
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
