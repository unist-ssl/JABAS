r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import time
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from iidp.optim.shard_optimizer import ShardSGD

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils
import argparse

import iidp

import jabas

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('data', metavar='DIR', nargs='?', default='data', help='path to dataset')
parser.add_argument('--data-dir', help='path to dataset')
parser.add_argument('--dataset', default='coco', help='dataset')
parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    help='images per gpu, the total batch size is $NGPU x batch_size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--epochs', default=13, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lr', default=0.02, type=float,
                    help='initial learning rate, 0.02 is the default value for training '
                    'on 8 gpus and 2 images_per_gpu')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--checkpoint-dir', default=None, help='path where to save')
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
parser.add_argument('--rpn-score-thresh', default=None, type=float,
                    help='rpn score threshold for faster-rcnn')
parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                    help='number of trainable layers of backbone')
parser.add_argument("--test-only", dest="test_only", action='store_true',
                    help="Only test the model")
parser.add_argument("--pretrained", dest="pretrained", action='store_true',
                    help="Use pre-trained models from the modelzoo")
parser.add_argument("--no-validate", action='store_true',
                    help='skip validation')
parser.add_argument("--num-minibatches", default=None, type=int,
                    help='number of minibatches')

# distributed training parameters
parser.add_argument("--multiprocessing-distributed", action='store_true',
                    help="multiprocessing distributed training")
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='rank number of distributed processes')
parser.add_argument('--dist-url', default='tcp://localhost:32000',
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='')

# Model summary
parser.add_argument('-v', '--verbose', default=None, type=int,
                    help='Verbose value of model summary')

# IIDP
parser.add_argument('--local-batch-size', '-lbs', default=2, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--accum-step', type=int, default=0, help='Gradient accumulation step')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--weight-sync-method', type=str, default='recommend',
                    choices=['recommend', 'overlap', 'sequential'],
                    help='Weight synchronization method in IIDP')

parser.add_argument('--jabas-config-file', type=str,
                help='JABAS training configuration file path (json)')
parser.add_argument('--elastic-checkpoint-dir', type=str, default=None,
                    help='checkpoint dir for elastic training')
parser.add_argument('--is-elastic-training', action='store_true',
                    help='Flag for elastic training')

parser.add_argument('--synthetic-dataset', action='store_true',
                    help='Synthetic dataset')


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for quick memory profile"""
    def __init__(self):
        self.length = 117266

    def __getitem__(self, index):
        # NOTE: Increase max shape with margin (1%)
        # to detect OOM by memory fragmentation as shape is dynamic in real trial
        MAX_DIM = 646
        MIN_DIM = int(MAX_DIM * 0.98)
        input_shape = [3, random.randint(MIN_DIM, MAX_DIM), random.randint(MIN_DIM, MAX_DIM)]
        target_size = 6
        target_shape = [target_size, input_shape[1], input_shape[2]]

        dummy_image = torch.randn(*input_shape)
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

        return dummy_image, dummy_target

    def __len__(self):
        return self.length


def main():
    args = parser.parse_args()

    if args.checkpoint_dir:
        utils.mkdir(args.checkpoint_dir)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.is_adaptive_training = (args.jabas_config_file is not None)
    # NOTE: Handle to positional argument 'data'
    if not args.is_elastic_training or args.data_dir is None:
        args.data_dir = args.data

    args.elastic_restart_timer = None
    if args.is_elastic_training:
        args.elastic_restart_timer = jabas.ElasticTrainReStartTimer(time.time())
        main_worker(0, 0, args)
    else:
        ngpus_per_node = torch.cuda.device_count()
        if args.multiprocessing_distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.is_elastic_training:
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        device = torch.device(args.gpu)
    else:
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
        device = torch.device(gpu)

    if args.distributed:
        if args.multiprocessing_distributed and not args.is_elastic_training:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else: # single-GPU
        args.rank = 0
        args.world_size = 1
        args.dist_url = 'tcp://127.0.0.1:22222'
        print(f'[INFO] single-GPU | args.rank: {args.rank}')
        print(f'[INFO] single-GPU | args.world_size: {args.world_size}')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print('[INFO] Local Batch Size:', args.local_batch_size)
    if args.is_adaptive_training:
        trainer = jabas.JABASTrainer(
            device, args.local_batch_size, args.num_models, args.accum_step,
            args.weight_sync_method, args.jabas_config_file, args.elastic_checkpoint_dir,
            args.elastic_restart_timer)
    else:
        trainer = iidp.IIDPTrainer(
            device, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method)
    args.batch_size = trainer.batch_size_per_gpu
    print('[INFO] Local Batch Size per GPU:', args.batch_size)
    args.global_batch_size = trainer.global_batch_size
    print('[INFO] Global Batch Size:', args.global_batch_size)

    # Data loading code
    print("[INFO] Loading data")

    if args.synthetic_dataset:
        dataset = SyntheticDataset()
        num_classes = 91
    else:
        dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_dir)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_dir)

    print(f'[INFO] num_classes: {num_classes}')
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    print(f"=> creating local model: {args.model}")
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=num_classes, pretrained=args.pretrained, **kwargs)
    model.to(device)

    class RCNNCriterion(object):
        def __init__(self):
            pass

        def __call__(self, loss_dict):
            # Output of model is loss's dict
            loss = sum(loss for loss in loss_dict.values())
            return loss

    criterion = RCNNCriterion()

    # Prepare local models to assign multi-streams
    trainer.prepare_stream_parallel(model, criterion)

    args.lr = scale_lr(args)
    print("[INFO] scaled lr:", args.lr)

    # Create local optimizer
    def build_param_groups_in_optimizer(model):
        return [p for p in model.parameters() if p.requires_grad]
    params = build_param_groups_in_optimizer(trainer.main_model)
    if trainer.weight_sync_method == 'overlap':
        optimizer = ShardSGD(params, args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    trainer.prepare_weight_sync_method(
        optimizer, lr_scheduler, param_groups_func=build_param_groups_in_optimizer)

    print("[INFO] Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = None
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if not args.synthetic_dataset and args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    def batch_func(batch, num_chunks, loading_once):
        if loading_once is True:
            def list_chunk(lst, num_elements):
                return [lst[i:i+num_elements] for i in range(0, len(lst), num_elements)]
            inputs, targets = batch[0], batch[1]
            inputs = list(input.to(device) for input in inputs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            chunked_inputs = list_chunk(inputs, args.local_batch_size)
            chunked_targets = list_chunk(targets, args.local_batch_size)
            parallel_local_data = []
            for chunked_input, chunked_target in zip(chunked_inputs, chunked_targets):
                if len(chunked_input) == 0 or len(chunked_target) == 0:
                    print(f'[WARNING] empty input or target: {chunked_input} | {chunked_target}')
                    print(f'[WARNING] inputs: {inputs.size()} | num_chunks: {num_chunks}')
                    return []
                parallel_local_data.append([chunked_input, chunked_target])
        else:
            parallel_local_data = []
            for (images, targets) in batch:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                parallel_local_data.append([images, targets])
        return parallel_local_data

    def size_func(batch):
        size = len(batch[0])
        return size

    if args.is_adaptive_training:
        train_loader = jabas.data.AdaptiveDataLoader(
            dataset, batch_fn=batch_func, size_fn=size_func, loading_once=False,
            batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)
        trainer.prepare_adaptive_data_loader(train_loader)
    else:
        train_loader = iidp.data.DataLoader(
            dataset, batch_fn=batch_func, loading_once=True,
            batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)
    train_sampler = train_loader.batch_sampler.sampler

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model = trainer.eval_model
    optimizer = trainer.main_optimizer
    lr_scheduler = trainer.main_scheduler

    if args.verbose:
        print(model)
        for _, param in enumerate(model.parameters()):
            if hasattr(param, 'index'):
                param_mem_value = round(param.nelement() * param.element_size() / (1024 ** 2), 3)
                print(f'[INFO] index: {param.index} | {param_mem_value} MB')

    if args.resume:
        trainer.load(args.resume)

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("[INFO] Start training! It takes some time to run.. ")

    if args.num_minibatches is not None:
        args.epochs = 1

    for epoch in trainer.remaining_epochs(args.epochs):
        if args.distributed and train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch < 8:
            trainer.config_params['batch_size_upper_bound'] = 64
        else:
            trainer.config_params['batch_size_upper_bound'] = 192
        with trainer.measure_epoch_time(), trainer.record_epoch_data():
            train_one_epoch(train_loader, trainer, epoch, args)
        trainer.scheduler_step()

        # evaluate after every epoch
        if args.no_validate:
            continue
        evaluate(model, data_loader_test, device=device)

        if args.checkpoint_dir and args.epochs > 1:
            if not args.distributed or \
                    (args.distributed and args.rank % ngpus_per_node == 0):
                trainer.save(args.checkpoint_dir)


def scale_lr(args):
    scaled_lr = args.lr * (args.global_batch_size / 16)
    return scaled_lr


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


if __name__ == "__main__":
    main()
