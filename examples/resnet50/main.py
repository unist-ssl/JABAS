import argparse
import os
import random
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import iidp
from iidp.optim import ShardSGD

import jabas

import numpy as np
import image_classification.resnet as models
from image_classification.mixup import NLLMultiLabelSmooth
from image_classification.smoothing import LabelSmoothing

model_names = models.resnet_versions.keys()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='data', help='path to dataset')
parser.add_argument('--data-dir', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--num-minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--no-validate', dest='no_validate', action='store_true',
                    help="No validation")
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--checkpoint-path', type=str,
                    default=None,
                    help='checkpoint path')

# simigrad original code
parser.add_argument("--lr-schedule", default="cosine", type=str, metavar="SCHEDULE",
                    choices=["step", "linear", "cosine"],
                    help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"))
parser.add_argument("--warmup", default=8, type=int, metavar="E", help="number of warmup epochs")
parser.add_argument("--label-smoothing", default=0.1, type=float, metavar="S", help="label smoothing")
parser.add_argument("--mixup", default=0.2, type=float, metavar="ALPHA", help="mixup alpha")
parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"],
                    help="memory layout, nchw or nhwc")

# IIDP
parser.add_argument('--local-batch-size', '-lbs', default=32, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--accum-step', type=int, default=0, help='Gradient accumulation step')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--weight-sync-method', type=str, default='recommend',
                    choices=['recommend', 'overlap', 'sequential'],
                    help='Weight synchronization method in IIDP')

parser.add_argument('--jabas-config-file', type=str,
                    help='Adaptive training configuration file path (json)')
parser.add_argument('--elastic-checkpoint-dir', type=str, default=None,
                    help='checkpoint dir for elastic training')
parser.add_argument('--is-elastic-training', action='store_true',
                    help='Flag for elastic training')


def lr_cosine_policy(warmup_length, epochs):
    INITIAL_GLOBAL_BATCH_SIZE = 256
    def _lr_fn(iteration, epoch):
        base_lr = 2.048 * math.sqrt(INITIAL_GLOBAL_BATCH_SIZE/2048)
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def main():
    args = parser.parse_args()

    if args.num_minibatches is not None:
        torch.cuda.empty_cache()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.is_adaptive_training = (args.jabas_config_file is not None)
    # NOTE: Handle to positional argument 'data'
    if args.is_elastic_training:
        args.data = args.data_dir

    args.elastic_restart_timer = None
    if args.is_elastic_training:
        args.elastic_restart_timer = jabas.ElasticTrainReStartTimer(time.time())
        main_worker(0, 0, args)
    else:
        ngpus_per_node = torch.cuda.device_count()
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # ======================================================================== #
    print(f'[INFO] Set up SimiGrad imagenet code parameters')
    args.lr = 2.048
    args.momentum = 0.875
    args.warmup = 8
    args.label_smoothing = 0.1
    args.weight_decay = 3.0517578125e-05
    args.model_config = 'fanin'
    args.num_classes = 1000
    args.mixup = 0.0
    # ======================================================================== #
    if args.is_elastic_training:
        local_rank = int(os.environ['JABAS_LOCAL_RANK'])
    else:
        args.gpu = gpu

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed and not args.is_elastic_training:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.dist_url = 'tcp://127.0.0.1:22222'
        print(f'[INFO] single-GPU | args.rank: {args.rank}')
        print(f'[INFO] single-GPU | args.world_size: {args.world_size}')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print('Local Batch Size:', args.local_batch_size)

    if args.is_adaptive_training:
        trainer = jabas.JABASTrainer(
            args.gpu, args.local_batch_size, args.num_models, args.accum_step,
            args.weight_sync_method, args.jabas_config_file, args.elastic_checkpoint_dir,
            args.elastic_restart_timer)
    else:
        trainer = iidp.IIDPTrainer(
            args.gpu, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method)

    args.batch_size = trainer.batch_size_per_gpu
    print('Local Batch Size per GPU:', args.batch_size)
    args.global_batch_size = trainer.global_batch_size
    print('Global Batch Size:', args.global_batch_size)
    args.lr = args.lr * math.sqrt(args.global_batch_size/2048)
    print('[IMPORTANT] Adjusted LR (default: 2.048 / GBS: 2048):', args.lr)

    # Create model
    model = models.build_resnet(args.arch, args.model_config, args.num_classes)
    model = model.to(args.gpu)

    # Define loss function (criterion) and optimizer
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
        criterion = loss().cuda(args.gpu)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)
        criterion = loss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Prepare stream parallelism
    trainer.prepare_stream_parallel(model, criterion)

    cudnn.benchmark = False

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    def chunk_func(batch, num_chunks, loading_once):
        if loading_once is True:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
            # NOTE: torch.chunk() may return smaller number of chunks
            chunked_inputs = torch.tensor_split(inputs, num_chunks)
            chunked_targets = torch.tensor_split(targets, num_chunks)
            parallel_local_data = []
            for chunked_input, chunked_target in zip(chunked_inputs, chunked_targets):
                if chunked_input.numel() == 0 or chunked_target.numel() == 0:
                    print(f'[WARNING] empty input or target: {chunked_input} | {chunked_target}')
                    print(f'[WARNING] inputs: {inputs.size()} | num_chunks: {num_chunks}')
                    return []
                parallel_local_data.append([chunked_input, chunked_target])
            return parallel_local_data
        else:
            parallel_local_data = []
            for (images, target) in batch:
                assert images.size()[0] == trainer.local_batch_size, \
                    f"Input size must be equal to local batch size, but {images.size()[0]} != {trainer.local_batch_size}"
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)
                parallel_local_data.append([images, target])
        return parallel_local_data

    def size_func(batch):
        size = batch[0].size()[0]
        return size

    if args.is_adaptive_training:
        train_loader = jabas.data.AdaptiveDataLoader(
            train_dataset, batch_size=args.batch_size, batch_fn=chunk_func, size_fn=size_func,
            loading_once=False, shuffle=(train_sampler is None), num_workers=args.workers,
            pin_memory=True, sampler=train_sampler)
        trainer.prepare_adaptive_data_loader(train_loader)
    else:
        train_loader = iidp.data.DataLoader(
            train_dataset, batch_size=args.batch_size, batch_fn=chunk_func, loading_once=True,
            shuffle=(train_sampler is None), num_workers=args.workers,
            pin_memory=True, sampler=train_sampler)

    validate_batch_size = 32
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=validate_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create optimizer
    def build_param_groups_in_optimizer(model):
        #print(f'[INFO] Set up SimiGrad optimizer - Weight decay NOT applied to BN parameters')
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if "bn" in n]
        rest_params = [v for n, v in parameters if not "bn" in n]
        optimizer_grouped_parameters  = [
            {"params": bn_params, "weight_decay": 0},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ]
        return optimizer_grouped_parameters

    optimizer_grouped_parameters = build_param_groups_in_optimizer(trainer.main_model)
    if trainer.weight_sync_method == 'overlap':
        optimizer = ShardSGD(optimizer_grouped_parameters, args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    print(f'[INFO] Set up SimiGrad LR scheduler - cosine with warmup of {args.warmup}')
    scheduler = lr_cosine_policy(args.warmup, args.epochs)

    trainer.prepare_weight_sync_method(optimizer, None, build_param_groups_in_optimizer)

    model = trainer.eval_model
    optimizer = trainer.main_optimizer

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load(args.resume)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.num_minibatches is not None:
        print('Number of mini-batches:', args.num_minibatches)
        args.epochs = 1
        print('Start epoch, epochs:', trainer.epoch, args.epochs)

    for epoch in trainer.remaining_epochs(args.epochs):
        if args.distributed and train_loader.sampler:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        with trainer.measure_epoch_time(), trainer.record_epoch_data():
            train(train_loader, trainer, epoch, scheduler, args)

        if not args.no_validate:
            # evaluate on validation set
            acc1 = validate(val_loader, trainer.eval_model, criterion, args)

        if args.checkpoint_path and args.epochs > 1:
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and \
                     args.rank % ngpus_per_node == 0):
                trainer.save(args.checkpoint_path)


def train(train_loader, trainer, epoch, scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    comp_time = AverageMeter('Comp', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')
    if args.is_adaptive_training:
        num_batches = len(train_loader.dataset)
    else:
        if args.num_minibatches is not None:
            num_batches = args.num_minibatches
        else:
            num_batches = len(train_loader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, comp_time, update_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    trainer.set_model_train()

    end = time.time()
    scheduler(trainer.main_optimizer, trainer.sync_step, epoch)
    is_sync_step = False
    warmup_step = 10 * (trainer.accum_step+1)
    for i, data in enumerate(train_loader):
        if args.num_minibatches is not None and trainer.sync_step > args.num_minibatches:
            break
        is_record_step = ((i+1) % (trainer.accum_step+1) == 0)
        # measure data loading time
        if i >= warmup_step and is_record_step:
            data_time.update(time.time() - end)
            start = time.time()

        if is_sync_step:
            scheduler(trainer.main_optimizer, trainer.sync_step, epoch)

        trainer.compute(data)

        # Record loss
        losses.update(trainer.losses[0].detach(), trainer.local_batch_size)

        if i >= warmup_step and is_record_step:
            comp_time.update(time.time() - start)
            start = time.time()

        # Update parameters
        is_sync_step = trainer.step()

        if i >= warmup_step and is_record_step:
            update_time.update(time.time() - start)

        # measure elapsed time
        if i >= warmup_step and is_record_step:
            batch_time.update(time.time() - end)
        if is_record_step:
            end = time.time()

        if args.is_adaptive_training:
            if is_record_step and (trainer.sync_step % args.print_freq == 0) and \
                    dist.get_rank() == 0:
                progress.display(train_loader.data_index)
        else:
            if is_record_step and ((train_loader.step_index+1) % args.print_freq == 0):
                # As step starts from 0, printing step+1 is right
                progress.display(train_loader.step_index+1)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.detach(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if dist.get_rank() == 0 and i % args.print_freq == 0:
                progress.display(i)

        if dist.get_rank() == 0:
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
