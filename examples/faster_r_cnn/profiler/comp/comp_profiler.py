import torch
import torchvision

from iidp.profiler import IIDPCustomProfilerHelper


class RCNNProfiler(IIDPCustomProfilerHelper):
    def __init__(self, lbs, num_models):
        super().__init__(lbs, num_models)

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

        self.prepare()

        self.auxiliary_profile_data = {'bn_sync_time': 14.048}

    def set_optimizer(self):
        def build_param_groups_in_optimizer(model):
            return [p for p in model.parameters() if p.requires_grad]
        model = self.trainer.main_model
        params = build_param_groups_in_optimizer(model)
        self.optimizer = torch.optim.SGD(params, 0.0001,
                                        momentum=0.9,
                                        weight_decay=0.004)
        self.param_groups_func = build_param_groups_in_optimizer

    def list_chunk(self, lst, num_elements):
        return [lst[i:i+num_elements] for i in range(0, len(lst), num_elements)]

    def run(self):
        print(f'====> Run IIDP profiler with the number of VSWs: {self.num_models}')
        self.trainer.set_model_train()
        batch_size = self.trainer.batch_size_per_gpu
        input_shape = [3, 424, 640]
        target_size = 6
        target_shape = [target_size, input_shape[1], input_shape[2]]
        for i in range(self.warmup_step+self.num_minibatches):
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
            with self.record_cuda_time():
                dummy_targets = [{k: v.to(self.gpu) for k, v in t.items()} for t in dummy_targets]
                scatter_images = self.list_chunk(dummy_images, self.trainer.local_batch_size)
                scatter_targets = self.list_chunk(dummy_targets, self.trainer.local_batch_size)
            data_time = self.cuda_time
            fwd_time, bwd_time = self.trainer.profile_parallel_compute(scatter_images, scatter_targets)
            update_time, copy_time = self.trainer.profile_step()
            if i >= self.warmup_step:
                total_time = data_time+ fwd_time + bwd_time + update_time + copy_time
                self.profile_data.update(data_time, fwd_time, bwd_time, update_time, copy_time, total_time)
                if i % 10 == 0:
                    print(f'[step {i}] {self.profile_data}')

        print(self.profile_data)
