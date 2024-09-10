import torch

from iidp.profiler import DDPHelper

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
import models # [IIDP] This must be imported for t2t_vit models

from timm.loss import SoftTargetCrossEntropy
from timm.models import create_model


class ViTProfiler(DDPHelper):
    def __init__(self):
        super().__init__()
        self.model_name = 'vit'
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
        self.model = create_model('t2t_vit_14', **model_config).to(self.gpu)
        self.criterion = SoftTargetCrossEntropy().to(self.gpu)
        self.lbs = 32 # Not important value

    def _get_ddp_bucket_indices(self):
        self.ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
            self.model, device_ids=[self.gpu], output_device=[self.gpu],
            model_index=0, num_local_models=1, total_num_models=1)
        self.ddp_module.train()
        input_shape = [self.lbs, 3, 224, 224]
        for _ in range(self.step):
            dummy_images = torch.randn(*input_shape).cuda(self.gpu, non_blocking=True)
            # NOTE: target shape is different from resnet50 on ImageNet
            dummy_targets = torch.randint(1000, size=(self.lbs, 1000),).cuda(self.gpu, non_blocking=True)

            output = self.ddp_module(dummy_images)
            loss = self.criterion(output, dummy_targets)
            loss.backward()

    def run(self):
        self.get_bucket_size_distribution()