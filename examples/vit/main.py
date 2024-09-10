# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT training and evaluating script
This script is modified from pytorch-image-models by Ross Wightman (https://github.com/rwightman/pytorch-image-models/)
It was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress

import models # [IIDP] This must be imported for t2t_vit models

import torch
import torch.nn as nn
from torch import optim as optim

from timm.data import Dataset, resolve_data_config, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.scheduler import create_scheduler

from data import create_loader, create_iidp_loader

import iidp
from iidp.optim import ShardAdamW

import jabas

torch.backends.cudnn.benchmark = False
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='T2T-ViT Training and Evaluating')

# Dataset / Model parameters
parser.add_argument('data', metavar='DIR', nargs='?', default='data', help='path to dataset')
parser.add_argument('--data-dir', help='path to dataset')
parser.add_argument('--model', default='T2t_vit_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.005 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=True,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')

parser.add_argument('--num-minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--no-validate', dest='no_validate', action='store_true',
                    help="No validation")

# IIDP
parser.add_argument('--local-batch-size', '-lbs', default=32, type=int,
                    help='Local batch size to be preserved')
parser.add_argument('--accum-step', type=int, default=0, help='Gradient accumulation step')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--weight-sync-method', type=str, default='recommend',
                    choices=['recommend', 'overlap', 'sequential'],
                    help='Weight synchronization method in IIDP')
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='checkpoint path')

parser.add_argument('--jabas-config-file', type=str,
                    help='Adaptive training configuration file path (json)')
parser.add_argument('--elastic-checkpoint-dir', type=str, default=None,
                    help='checkpoint dir for elastic training')
parser.add_argument('--is-elastic-training', action='store_true',
                    help='Flag for elastic training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.channels_last or args.no_prefetcher or args.clip_grad:
        raise ValueError(
            f'Not support '
            f'--channels-last or --no-prefetcher or --clip-grad '
            f'for IIDP'
        )

    args.prefetcher = not args.no_prefetcher

    args.is_adaptive_training = (args.jabas_config_file is not None)
    elastic_restart_timer = None
    if args.is_elastic_training:
        args.data = args.data_dir
        elastic_restart_timer = jabas.ElasticTrainReStartTimer(time.time())
        args.distributed = True
        args.local_rank = int(os.environ['JABAS_LOCAL_RANK'])
    else:
        args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        if args.is_elastic_training:
            torch.distributed.init_process_group(
                    backend=args.dist_backend, init_method=args.dist_url,
                    world_size=args.world_size, rank=args.rank)
        else:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
    else:
        args.device = 'cuda:0'
        args.rank = 0
        args.world_size = 1
        args.dist_url = 'tcp://127.0.0.1:22222'
        print(f'[INFO] single-GPU | args.rank: {args.rank}')
        print(f'[INFO] single-GPU | args.world_size: {args.world_size}')
        torch.distributed.init_process_group(
                backend='nccl', init_method=args.dist_url,
                world_size=args.world_size, rank=args.rank)
    assert args.rank >= 0

    if args.distributed:
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    print('Local Batch Size:', args.local_batch_size)
    if args.is_adaptive_training:
        trainer = jabas.JABASTrainer(
            args.local_rank, args.local_batch_size, args.num_models, args.accum_step,
            args.weight_sync_method, args.jabas_config_file, args.elastic_checkpoint_dir,
            elastic_restart_timer)
    else:
        trainer = iidp.IIDPTrainer(
            args.local_rank, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method)

    args.batch_size = trainer.batch_size_per_gpu
    print('Local Batch Size per GPU:', args.batch_size)
    args.global_batch_size = trainer.global_batch_size
    print('Global Batch Size:', args.global_batch_size)

    # Batch size constraint
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        assert args.batch_size % 2 == 0, "With mix-up, --batch-size % 2 == 0"

    local_models = []
    for i in range(trainer.num_local_models):
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=False,
            bn_momentum=None,
            bn_eps=None,
            checkpoint_path=args.initial_checkpoint,
            img_size=args.img_size)
        model.cuda()
        local_models.append(model)
        if i == 0:
            if args.local_rank == 0:
                print('Model %s created, param count: %d' %
                            (args.model, sum([m.numel() for m in model.parameters()])))

            data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    trainer.set_original_local_models(local_models)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # Define loss function (criterion)
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    if args.local_rank == 0:
        print(f'[INFO] criterion: {type(train_loss_fn)}')
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # Prepare stream parallelism
    trainer.prepare_stream_parallel(model, train_loss_fn)
    model = trainer.eval_model

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
            {'params': decay, 'weight_decay': args.weight_decay}
        ]
        args.weight_decay = 0.
        return optimizer_grouped_parameters

    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    optimizer_grouped_parameters = build_param_groups_in_optimizer(model)
    if trainer.weight_sync_method == 'overlap':
        optimizer = ShardAdamW(optimizer_grouped_parameters, **opt_args)
    else:
        optimizer = optim.AdamW(optimizer_grouped_parameters, **opt_args)

    amp_autocast = suppress  # do nothing
    loss_scaler = None

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load(args.resume)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    if args.num_minibatches is not None:
        print('Number of mini-batches:', args.num_minibatches)
        start_epoch = 0
        args.epochs = 1
        print('Start epoch, epochs:', start_epoch, args.epochs)

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch

    if lr_scheduler is not None and trainer.epoch > 0:
        lr_scheduler.step(trainer.epoch)

    if args.epochs != 300: # 300 is default
        orig_num_epochs = num_epochs
        num_epochs = args.epochs
    if args.local_rank == 0:
        print('Scheduled epochs: {}'.format(num_epochs))

    trainer.prepare_weight_sync_method(optimizer, lr_scheduler, param_groups_func=build_param_groups_in_optimizer)
    optimizer = trainer.main_optimizer
    lr_scheduler = trainer.main_scheduler

    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_iidp_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        is_adaptive_training=args.is_adaptive_training
    )
    if args.is_adaptive_training:
        trainer.prepare_adaptive_data_loader(loader_train)

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=32,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    if args.eval_checkpoint:  # evaluate the model
        load_checkpoint(model, args.eval_checkpoint, args.model_ema)
        val_metrics = validate(model, loader_eval, validate_loss_fn, args)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        return

    try:  # train the model
        for epoch in trainer.remaining_epochs(args.epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            with trainer.measure_epoch_time(), trainer.record_epoch_data():
                train_metrics = train(loader_train, trainer, epoch, args, model_ema=model_ema)
            if args.no_validate:
                break

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.checkpoint_path and args.epochs > 1:
                if args.rank == 0:
                    trainer.save(args.checkpoint_path)

    except KeyboardInterrupt:
        pass
    if best_metric is not None and args.local_rank == 0:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train(train_loader, trainer, epoch, args, model_ema=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    comp_time = AverageMeter('Comp', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    ema_time = AverageMeter('EMA', ':6.5f')
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
        [batch_time, data_time, comp_time, update_time, ema_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    trainer.set_model_train()

    end = time.time()
    num_updates = epoch * len(train_loader)
    warmup_step = 10 * (trainer.accum_step+1)
    for batch_idx, data in enumerate(train_loader):
        if args.num_minibatches is not None and trainer.sync_step > args.num_minibatches:
            break
        is_record_step = ((batch_idx+1) % (trainer.accum_step+1) == 0)
        # measure data loading time
        if batch_idx >= warmup_step and is_record_step:
            data_time.update(time.time() - end)
            start = time.time()

        trainer.compute(data)
        # Record loss
        losses.update(trainer.losses[0].detach(), trainer.local_batch_size)
        if batch_idx >= warmup_step and is_record_step:
            comp_time.update(time.time() - start)
            start = time.time()
        # Update parameters
        is_sync_step = trainer.step()
        if batch_idx >= warmup_step and is_record_step:
            update_time.update(time.time() - start)
            start = time.time()
        if model_ema is not None and is_sync_step:
            model_ema.update(trainer.main_model)
        if batch_idx >= warmup_step and is_record_step:
            ema_time.update(time.time() - start)

        if args.is_adaptive_training:
            num_updates = int((train_loader.get_progress() / len(train_loader.dataset)) * len(train_loader))
        else:
            if is_sync_step:
                num_updates += 1
        if trainer.main_scheduler is not None:
            trainer.main_scheduler.step_update(num_updates=trainer.sync_step, metric=losses.avg)

        if batch_idx >= warmup_step and is_record_step:
            batch_time.update(time.time() - end)

        if is_record_step:
            end = time.time()

        if args.is_adaptive_training:
            if is_record_step and (trainer.sync_step % args.log_interval == 0) and \
                    torch.distributed.get_rank() == 0:
                progress.display(train_loader.data_index)
        else:
            if is_record_step and ((train_loader.step_index+1) % args.log_interval == 0):
                # As step starts from 0, printing step+1 is right
                progress.display(train_loader.step_index+1)

    return OrderedDict([('loss', losses.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter('Time', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')
    top1_m = AverageMeter('Acc@1', '.:4f')
    top5_m = AverageMeter('Acc@5', '.:4f')

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    if args.local_rank == 0 and not (log_suffix == ' (EMA)'):
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


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


if __name__ == '__main__':
    main()
