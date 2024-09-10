import torch

from iidp.profiler import DDPHelper, CUDAEventTimer

import os, sys ; sys.path.append(os.path.dirname(__file__) +'/../..')
import image_classification.resnet as models
from image_classification.smoothing import LabelSmoothing


class ResNet50Profiler(DDPHelper):
    def __init__(self):
        super().__init__()
        self.model_name = 'resnet50'
        self.model_config = 'fanin'
        self.num_classes = 1000
        self.label_smoothing = 0.1
        self.model = models.build_resnet(self.model_name, self.model_config, self.num_classes).to(self.gpu)
        loss = lambda: LabelSmoothing(self.label_smoothing)
        self.criterion = loss().to(self.gpu)
        self.lbs = 32 # Not important value

    def _get_ddp_bucket_indices(self):
        self.ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
            self.model, device_ids=[self.gpu], output_device=[self.gpu],
            model_index=0, num_local_models=1, total_num_models=1)
        self.ddp_module.train()
        input_shape = [self.lbs, 3, 224, 224]
        print(f'[DDPHelper] step: {self.step}')
        for step in range(self.step):
            dummy_images = torch.randn(*input_shape).cuda(self.gpu, non_blocking=True)
            dummy_targets = torch.empty(self.lbs, dtype=torch.int64).random_(1000).cuda(self.gpu, non_blocking=True)
            is_verbose = (step >= 1)
            with CUDAEventTimer('[Profile info] DDP forward (+BN sync) time', verbose=is_verbose) as timer:
                output = self.ddp_module(dummy_images)
            loss = self.criterion(output, dummy_targets)
            loss.backward()

    def run(self):
        self.get_bucket_size_distribution()