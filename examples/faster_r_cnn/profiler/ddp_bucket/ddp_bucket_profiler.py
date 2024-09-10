import torch
import torchvision

from iidp.profiler import DDPHelper, CUDAEventTimer


class RCNNProfiler(DDPHelper):
    def __init__(self):
        super().__init__()
        self.model_name = 'rcnn'

        num_classes = 91
        pretrained = False
        model = 'fasterrcnn_resnet50_fpn'
        kwargs = {
            "trainable_backbone_layers": None
        }
        self.model = torchvision.models.detection.__dict__[model](
            num_classes=num_classes, pretrained=pretrained, **kwargs).to(self.gpu)

        class RCNNCriterion(object):
            def __init__(self):
                pass

            def __call__(self, loss_dict):
                # Output of model is loss's dict
                loss = sum(loss for loss in loss_dict.values())
                return loss
        self.criterion = RCNNCriterion()
        self.lbs = 2

    def _get_ddp_bucket_indices(self):
        self.ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
            self.model, device_ids=[self.gpu], output_device=[self.gpu],
            model_index=0, num_local_models=1, total_num_models=1)
        self.ddp_module.train()

        batch_size = self.lbs
        input_shape = [3, 424, 640]
        target_size = 6
        target_shape = [target_size, input_shape[1], input_shape[2]]
        for step in range(self.step):
            dummy_image = torch.randn(*input_shape).to(self.gpu)
            boxes = 300*torch.rand([target_size, 4])
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=input_shape[-2])
            boxes[:, 1::2].clamp_(min=0, max=input_shape[-1])
            dummy_target = {
                # boxes (Tensor[N, 4]) in vision/torchvision/ops/boxes.py
                'boxes': boxes,
                'labels': torch.randint(high=91, size=(target_size,)),
                'masks': torch.zeros(*target_shape),
                'image_id': torch.tensor([244973]),
                'area': torch.randint(high=50000, size=(target_size,), dtype=torch.float),
                'iscrowd': torch.zeros(target_size)
            }
            dummy_images, dummy_targets = [], []
            for _ in range(batch_size):
                dummy_images.append(dummy_image)
                dummy_targets.append(dummy_target)
            dummy_targets = [{k: v.to(self.gpu) for k, v in t.items()} for t in dummy_targets]

            is_verbose = (step >= 1)
            with CUDAEventTimer('[Profile info] DDP forward (+BN sync) time', verbose=is_verbose) as timer:
                loss = self.criterion(self.ddp_module(dummy_images, dummy_targets))
            loss.backward()

    def run(self):
        self.get_bucket_size_distribution()