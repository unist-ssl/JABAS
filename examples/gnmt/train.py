#!/usr/bin/env python

# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import os
import sys
import time
import math
import warnings
from ast import literal_eval

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from seq2seq.train.shard_optimizer import ShardGNMTAdam

import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
import seq2seq.utils as utils
from seq2seq.data.dataset import LazyParallelDataset
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.dataset import TextDataset
from seq2seq.data.dataset import SequenceCollateFunction
from seq2seq.data.sampler import BucketingImbalancedSampler
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.translator import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.train.table import TrainingTable
from seq2seq.train.lr_scheduler import WarmupMultiStepLR, AdaptiveWarmupMultiStepLR

import iidp

import jabas


best_loss = float('inf')


def parse_args():
    """
    Parse commandline arguments.
    """
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')
    dataset.add_argument('--data-dir', default='data/wmt16_de_en',
                         help='path to the directory with training/test data')

    dataset.add_argument('--src-lang',
                         default='en',
                         help='source language')
    dataset.add_argument('--tgt-lang',
                         default='de',
                         help='target language')

    dataset.add_argument('--vocab',
                         default='vocab.bpe.32000',
                         help='path to the vocabulary file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('-bpe', '--bpe-codes', default='bpe.32000',
                         help='path to the file with bpe codes \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--train-src',
                         default='train.tok.clean.bpe.32000.en',
                         help='path to the training source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--train-tgt',
                         default='train.tok.clean.bpe.32000.de',
                         help='path to the training target data file \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--val-src',
                         default='newstest_dev.tok.clean.bpe.32000.en',
                         help='path to the validation source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--val-tgt',
                         default='newstest_dev.tok.clean.bpe.32000.de',
                         help='path to the validation target data file \
                         (relative to DATASET_DIR directory)')

    dataset.add_argument('--test-src',
                         default='newstest2014.tok.bpe.32000.en',
                         help='path to the test source data file \
                         (relative to DATASET_DIR directory)')
    dataset.add_argument('--test-tgt',
                         default='newstest2014.de',
                         help='path to the test target data file \
                         (relative to DATASET_DIR directory)')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--log-dir', default=None,
                         help='defines directory for log file from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--hidden-size', default=1024, type=int,
                       help='hidden size of the model')
    model.add_argument('--num-layers', default=4, type=int,
                       help='number of RNN layers in encoder and in decoder')
    model.add_argument('--dropout', default=0.2, type=float,
                       help='dropout applied to input of RNN cells')

    exclusive_group(group=model, name='share-embedding', default=True,
                    help='use shared embeddings for encoder and decoder')

    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                       CrossEntropyLoss, if not zero model will be trained \
                       with label smoothing loss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp32',
                         choices=['fp32'],
                         help='mixed precision not supported.')
    general.add_argument('--seed', default=None, type=int,
                         help='master seed for random number generators, if \
                         "seed" is undefined then the master seed will be \
                         sampled from random.SystemRandom()')

    exclusive_group(group=general, name='eval', default=True,
                    help='run validation and test after every epoch')
    exclusive_group(group=general, name='env', default=True,
                    help='print info about execution env')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')
    exclusive_group(group=general, name='log-all-ranks', default=True,
                    help='enables logging from all distributed ranks, if \
                    disabled then only logs from rank 0 are reported')

    # training
    training = parser.add_argument_group('training setup')
    dataset.add_argument('--train-max-size', default=None, type=int,
                         help='use at most TRAIN_MAX_SIZE elements from \
                         training dataset (useful for benchmarking), by \
                         default uses entire dataset')
    training.add_argument('--train-batch-size', default=128, type=int,
                          help='training batch size per worker')
    training.add_argument('--train-global-batch-size', default=None, type=int,
                          help='global training batch size, this argument \
                          does not have to be defined, if it is defined it \
                          will be used to automatically \
                          compute train_iter_size \
                          using the equation: train_iter_size = \
                          train_global_batch_size // (train_batch_size * \
                          world_size)')
    training.add_argument('--train-iter-size', metavar='N', default=1,
                          type=int,
                          help='training iter size, training loop will \
                          accumulate gradients over N iterations and execute \
                          optimizer every N steps')
    training.add_argument('--epochs', default=6, type=int,
                          help='max number of training epochs')

    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enables gradient clipping and sets maximum \
                          norm of gradients')
    training.add_argument('--train-max-length', default=50, type=int,
                          help='maximum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--train-min-length', default=0, type=int,
                          help='minimum sequence length for training \
                          (including special BOS and EOS tokens)')
    training.add_argument('--batching', default='bucketing', type=str,
                          choices=['random', 'sharding', 'bucketing'],
                          help='select batching algorithm')
    training.add_argument('--shard-size', default=80, type=int,
                          help='shard size for "sharding" batching algorithm, \
                          in multiples of global batch size')
    training.add_argument('--num-buckets', default=5, type=int,
                          help='number of buckets for "bucketing" batching \
                          algorithm')

    # optimizer
    optimizer = parser.add_argument_group('optimizer setup')
    optimizer.add_argument('--optimizer', type=str, default='Adam',
                           help='training optimizer')
    optimizer.add_argument('--lr', type=float, default=2.00e-3,
                           help='learning rate')
    optimizer.add_argument('--optimizer-extra', type=str,
                           default="{}",
                           help='extra options for the optimizer')

    # mixed precision loss scaling
    loss_scaling = parser.add_argument_group(
        'mixed precision loss scaling setup'
        )
    loss_scaling.add_argument('--init-scale', type=float, default=8192,
                              help='initial loss scale')
    loss_scaling.add_argument('--upscale-interval', type=float, default=128,
                              help='loss upscaling interval')

    # scheduler
    scheduler = parser.add_argument_group('learning rate scheduler setup')
    scheduler.add_argument('--warmup-steps', type=str, default='200',
                           help='number of learning rate warmup iterations')
    scheduler.add_argument('--remain-steps', type=str, default='0.666',
                           help='starting iteration for learning rate decay')
    scheduler.add_argument('--decay-interval', type=str, default='None',
                           help='interval between learning rate decay steps')
    scheduler.add_argument('--decay-steps', type=int, default=4,
                           help='max number of learning rate decay steps')
    scheduler.add_argument('--decay-factor', type=float, default=0.5,
                           help='learning rate decay factor')

    # validation
    val = parser.add_argument_group('validation setup')
    val.add_argument('--val-batch-size', default=64, type=int,
                     help='batch size for validation')
    val.add_argument('--val-max-length', default=125, type=int,
                     help='maximum sequence length for validation \
                     (including special BOS and EOS tokens)')
    val.add_argument('--val-min-length', default=0, type=int,
                     help='minimum sequence length for validation \
                     (including special BOS and EOS tokens)')

    # test
    test = parser.add_argument_group('test setup')
    test.add_argument('--test-batch-size', default=128, type=int,
                      help='batch size for test')
    test.add_argument('--test-max-length', default=150, type=int,
                      help='maximum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--test-min-length', default=0, type=int,
                      help='minimum sequence length for test \
                      (including special BOS and EOS tokens)')
    test.add_argument('--beam-size', default=5, type=int,
                      help='beam size')
    test.add_argument('--len-norm-factor', default=0.6, type=float,
                      help='length normalization factor')
    test.add_argument('--cov-penalty-factor', default=0.1, type=float,
                      help='coverage penalty factor')
    test.add_argument('--len-norm-const', default=5.0, type=float,
                      help='length normalization constant')
    test.add_argument('--intra-epoch-eval', metavar='N', default=0, type=int,
                      help='evaluate within training epoch, this option will \
                      enable extra N equally spaced evaluations executed \
                      during each training epoch')

    # checkpointing
    chkpt = parser.add_argument_group('checkpointing setup')
    chkpt.add_argument('--start-epoch', default=0, type=int,
                       help='manually set initial epoch counter')
    chkpt.add_argument('--resume', default=None, type=str, metavar='PATH',
                       help='resumes training from checkpoint from PATH')
    chkpt.add_argument('--save-dir', default=None,
                         help='defines directory for checkpoint \
                         results from this training run. \
                         If None, checkpointing is not saved')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-perf', default=None, type=float,
                           help='target training performance (in tokens \
                           per second)')
    benchmark.add_argument('--target-bleu', default=None, type=float,
                           help='target accuracy')

    # multi-node training
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training. This should be'
                             'the IP address and open port number of the master node')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--num-minibatches', default=None, type=int,
                        help='Restrict batch iteration.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help="multiprocessing distributed training")

    # single GPU
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # evaluation
    parser.add_argument('--eval-dir', default=None, type=str,
                        help='Directory path for evaluation. '
                             'If eval is ture, it must be configured.')

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
    parser.add_argument('--dummy-input', action='store_true',
                        help='Flag for dummy input for throughput measurement')

    args = parser.parse_args()

    args.lang = {'src': args.src_lang, 'tgt': args.tgt_lang}

    args.vocab = os.path.join(args.dataset_dir, args.vocab)
    args.bpe_codes = os.path.join(args.dataset_dir, args.bpe_codes)
    args.train_src = os.path.join(args.dataset_dir, args.train_src)
    args.train_tgt = os.path.join(args.dataset_dir, args.train_tgt)
    args.val_src = os.path.join(args.dataset_dir, args.val_src)
    args.val_tgt = os.path.join(args.dataset_dir, args.val_tgt)
    args.test_src = os.path.join(args.dataset_dir, args.test_src)
    args.test_tgt = os.path.join(args.dataset_dir, args.test_tgt)

    args.warmup_steps = literal_eval(args.warmup_steps)
    args.remain_steps = literal_eval(args.remain_steps)
    args.decay_interval = literal_eval(args.decay_interval)

    return args


def main():
    args = parse_args()

    ############################################################
    args.is_adaptive_training = (args.jabas_config_file is not None)

    args.elastic_restart_timer = None
    if args.is_elastic_training:
        args.distributed = True
        args.dataset_dir = args.data_dir
        args.elastic_restart_timer = jabas.ElasticTrainReStartTimer(time.time())
        main_worker(0, 0, args)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size
        args.distributed = args.world_size > 1 or args.multiprocessing_distributed
        if args.distributed:
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            print('[INFO] ngpus_per_node:', ngpus_per_node)
            print('[INFO] args.world_size:', args.world_size)
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)
    ############################################################


def build_criterion(padding_idx, smoothing, local_batch_size):
    if smoothing == 0.:
        logging.info(f'Building CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        logging.info(f'Building LabelSmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing, local_batch_size)

    return criterion


def main_worker(gpu, ngpus_per_node, args):
    """
    :param gpu: this is the process index. mp.spawn will assign it
                from 0 to ngpus - 1 for the current node
    :param ngpus_per_node:
    :param args:
    :return:
    """
    global best_loss
    if args.gpu is not None: # Single-GPU or Elastic training
        device = utils.set_device(args.cuda, args.gpu)
        print("[INFO] Use GPU: {} for training".format(args.gpu))
    else:
        device = utils.set_device(args.cuda, gpu)
        print("[INFO] Use GPU: {} for training".format(gpu))

    if args.distributed:
        ############################################################
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes across all nodes
        if not args.is_elastic_training:
            args.rank = args.rank * ngpus_per_node + gpu
        # No process will continue until all processes have joined.
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        ############################################################
    else:
        args.rank = 0
        args.world_size = 1
        args.dist_url = 'tcp://127.0.0.1:22222'
        print(f'[INFO] single-GPU | args.rank: {args.rank}')
        print(f'[INFO] single-GPU | args.world_size: {args.world_size}')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # cudnn will look for the optimal set of algorithms for that
    # particular configuration. this will have faster runtime if
    # your input sizes does not change at each iteration
    cudnn.benchmark = False

    if args.save_dir:
        args.is_save = True
    else:
        args.is_save = False

    if args.num_minibatches is not None:
        args.epochs = 1
        args.eval = False
        print('[INFO] If args.num_minibatches is configured, '
              'args.epoch = 1 and args.eval = False')

    if args.eval and not args.eval_dir:
        raise AssertionError(
            "If evaluation mode, --eval-dir must be configured to save translation")

    # create directory for results
    # currently, raise an error if directory alreay exists
    if args.rank == 0:
        if args.is_elastic_training is True:
            exist_ok = True
        else:
            exist_ok = False
        if args.log_dir:
            print(f'[INFO] args.log_dir: {args.log_dir}')
            os.makedirs(args.log_dir, exist_ok=exist_ok)
        if args.save_dir and not args.resume:
            print(f'[INFO] args.save_dir for checkpointing: {args.save_dir}')
            os.makedirs(args.save_dir, exist_ok=exist_ok)
        if args.eval_dir:
            print(f'[INFO] args.eval_dir for evaluation: {args.eval_dir}')
            os.makedirs(args.eval_dir, exist_ok=exist_ok)

    # setup logging
    print('[INFO] utils.get_rank():', utils.get_rank())
    if args.log_dir:
        log_filename = f'log_rank_{utils.get_rank()}.log'
        log_filepath = os.path.join(args.log_dir, log_filename)
        logging.info(f'Saving log results to: {args.log_dir}')
    else:
        log_filepath = None
    utils.setup_logging(args.log_all_ranks, log_filepath)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logging.info(f'Run arguments: {args}')

    # Not support gradient accumulation
    args.train_iter_size = 1

    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed,
                                                      args.epochs,
                                                      device)
    worker_seed = worker_seeds[args.rank]
    logging.info(f'Worker {args.rank} is using worker seed: {worker_seed}')
    torch.manual_seed(worker_seed)

    print('[INFO] Local Batch Size:', args.local_batch_size)

    # Create IIDP Trainer
    if args.is_adaptive_training:
        dp_trainer = jabas.JABASTrainer(
            device, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method,
            args.jabas_config_file, args.elastic_checkpoint_dir, args.elastic_restart_timer)
    else:
        dp_trainer = iidp.IIDPTrainer(
            device, args.local_batch_size, args.num_models, args.accum_step, args.weight_sync_method)

    args.train_batch_size = dp_trainer.batch_size_per_gpu
    print('[INFO] Local Batch Size per GPU:', args.train_batch_size)
    args.train_global_batch_size = dp_trainer.global_batch_size

    # build tokenizer
    pad_vocab = utils.pad_vocabulary(args.math)
    tokenizer = Tokenizer(args.vocab, args.bpe_codes, args.lang, pad_vocab)

    # build datasets
    train_data = LazyParallelDataset(
        src_fname=args.train_src,
        tgt_fname=args.train_tgt,
        tokenizer=tokenizer,
        min_len=args.train_min_length,
        max_len=args.train_max_length,
        sort=False,
        max_size=args.train_max_size,
        )

    val_data = ParallelDataset(
        src_fname=args.val_src,
        tgt_fname=args.val_tgt,
        tokenizer=tokenizer,
        min_len=args.val_min_length,
        max_len=args.val_max_length,
        sort=True,
        )

    test_data = TextDataset(
        src_fname=args.test_src,
        tokenizer=tokenizer,
        min_len=args.test_min_length,
        max_len=args.test_max_length,
        sort=True,
        )

    vocab_size = tokenizer.vocab_size

    # build local GNMT model
    model_config = {'hidden_size': args.hidden_size,
                    'vocab_size': vocab_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'batch_first': False,
                    'share_embedding': args.share_embedding,
                    }
    """
    # Issue: https://discuss.pytorch.org/t/copy-deepcopy-or-clone-cause-userwarning/78749/3
    print(f"=> creating local model: gnmt")
    model = GNMT(**model_config).to(device)
    logging.info(model)
    batch_first = model.batch_first
    """
    local_models = []
    for _ in range(dp_trainer.num_local_models):
        model =  GNMT(**model_config).to(device)
        local_models.append(model)
    dp_trainer.set_original_local_models(local_models)
    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(
        config.PAD, args.smoothing, batch_first).to(device)

    # Prepare stream parallelism
    dp_trainer.prepare_stream_parallel(model, criterion, gradient_as_bucket_view=True)

    # LR scaling rule
    gnmt_optimal_lr = args.lr
    scale = args.train_global_batch_size / 1024
    if args.train_global_batch_size < 1024:
        # [EXPERIMENTAL]
        if args.train_global_batch_size == 512:
            gnmt_optimal_lr = args.lr * scale
        elif args.train_global_batch_size == 128:
            gnmt_optimal_lr = args.lr * math.sqrt(scale)
        else:
            if not args.is_elastic_training:
                if args.num_minibatches is None: # Convergence
                    raise ValueError(f'[EXPERIMENTAL] Not support initial global batch size: '
                                    f'{args.train_global_batch_size} with scaled learning rate')
                else:
                    warnings.warn(f'[EXPERIMENTAL] Not confirm to achieve target BLEU with '
                                f'initial global batch size: **{args.train_global_batch_size}**')
            else:
                # TODO: with elastic training (ckpt-based restart), check initial LR at epoch 0 only.
                pass
    else:
        # LEGW (Linear Epoch Gradual Warmup)
        # Increase base lr by square-root of k times and warmup epochs by k times
        # Paper: Large-Batch Training for LSTM and Beyond (Link: https://arxiv.org/pdf/1901.08256.pdf)
        gnmt_optimal_lr = args.lr * math.sqrt(scale)
        args.warmup_steps = int(args.warmup_steps * scale)
    opt_config = {'lr': gnmt_optimal_lr}
    opt_config.update(literal_eval(args.optimizer_extra))
    logging.info(f'Training optimizer config: {opt_config}')

    scheduler_config = {'warmup_steps': args.warmup_steps,
                        'remain_steps': args.remain_steps,
                        'decay_interval': args.decay_interval,
                        'decay_steps': args.decay_steps,
                        'decay_factor': args.decay_factor}

    logging.info(f'Training LR schedule config: {scheduler_config}')

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    batching_opt = {'shard_size': args.shard_size,
                    'num_buckets': args.num_buckets}

    # get train data loaders
    collate_fn = SequenceCollateFunction(batch_first, parallel=True, sort=True)
    if args.batching == 'bucketing':
        train_sampler = BucketingImbalancedSampler(train_data, args.train_batch_size,
            shuffling_seeds, batching_opt['num_buckets'], args.train_global_batch_size,
            dp_trainer.all_partition_size_in_process_group)
    else:
        raise NotImplementedError

    def batch_func(batch, num_chunks, loading_once):
        num_toks = {}
        if loading_once:
            def _scatter_sentence(sentence, chunk_num):
                tgt1, tgt2 = sentence
                tgt1s = torch.tensor_split(tgt1, chunk_num, dim=1)
                tgt2s = torch.tensor_split(tgt2, chunk_num, dim=0)
                result = list(zip(tgt1s, tgt2s))
                return result

            def _generate_scatter_inputs_and_targets(src, tgt):
                scatter_inputs, scatter_targets = [], []

                scatter_srcs = _scatter_sentence(src, num_chunks)
                scatter_tgts = _scatter_sentence(tgt, num_chunks)

                for scatter_src, scatter_tgt in zip(scatter_srcs, scatter_tgts):
                    scatter_src, src_length = scatter_src
                    scatter_tgt, _ = scatter_tgt

                    scatter_src = scatter_src.to(device)
                    src_length = src_length.to(device)
                    scatter_tgt = scatter_tgt.to(device)

                    if batch_first:
                        scatter_input = [scatter_src, src_length, scatter_tgt[:, :-1]]
                        scatter_target = scatter_tgt[:, 1:].contiguous().view(-1)
                    else:
                        scatter_input = [scatter_src, src_length, scatter_tgt[:-1]]
                        scatter_target = scatter_tgt[1:].contiguous().view(-1)
                    scatter_inputs.append(scatter_input)
                    scatter_targets.append(scatter_target)
                return scatter_inputs, scatter_targets
            src, tgt = batch
            num_toks['src'] = int(sum(src[-1])) // num_chunks
            num_toks['tgt'] = int(sum(tgt[-1]-1)) // num_chunks
            chunked_inputs, chunked_targets = _generate_scatter_inputs_and_targets(src, tgt)
            parallel_local_data = []
            for chunked_input, chunked_target in zip(chunked_inputs, chunked_targets):
                if len(chunked_input) == 0 or len(chunked_target) == 0:
                    print(f'[WARNING] empty input or target: {chunked_input} | {chunked_target}')
                    print(f'[WARNING] inputs: {src.size()} | num_chunks: {num_chunks}')
                    return []
                parallel_local_data.append([chunked_input, chunked_target])
        else:
            def _prepare_local_batch_data(src, tgt):
                if args.dummy_input:
                    batch_size = src[0].size()[1]
                    if batch_first:
                        src = torch.ones(batch_size, 50, dtype=torch.int64)
                        src_length = torch.ones(batch_size, dtype=torch.int64)
                        tgt = torch.ones(batch_size, 50, dtype=torch.int64)
                    else:
                        src = torch.ones(50, batch_size, dtype=torch.int64)
                        src_length = torch.ones(batch_size, dtype=torch.int64)
                        tgt = torch.ones(50, batch_size, dtype=torch.int64)
                else:
                    src, src_length = src
                    tgt, _ = tgt

                src = src.to(device)
                src_length = src_length.to(device)
                tgt = tgt.to(device)
                if batch_first:
                    parallel_input = [src, src_length, tgt[:, :-1]]
                    parallel_target = tgt[:, 1:].contiguous().view(-1)
                else:
                    parallel_input = [src, src_length, tgt[:-1]]
                    parallel_target = tgt[1:].contiguous().view(-1)

                return parallel_input, parallel_target
            parallel_local_data = []
            tot_num_src_toks, tot_num_tgt_toks = 0, 0
            for (src, tgt) in batch:
                tot_num_src_toks += int(sum(src[-1]))
                tot_num_tgt_toks += int(sum(tgt[-1]-1))
                parallel_local_data.append(_prepare_local_batch_data(src, tgt))
            num_toks['src'] = tot_num_src_toks / num_chunks
            num_toks['tgt'] = tot_num_tgt_toks / num_chunks
        return parallel_local_data, num_toks

    def size_func(batch):
        size = batch[0][0].size()[1]
        return size

    if args.dummy_input:
        logging.info('[EXPERIMENT] dummy static shape input is configured')

    if args.is_adaptive_training:
        train_loader = jabas.data.AdaptiveDataLoader(
            train_data, batch_size=args.train_batch_size, batch_fn=batch_func, size_fn=size_func,
            loading_once=False, collate_fn=collate_fn, sampler=train_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=False)
        dp_trainer.prepare_adaptive_data_loader(train_loader)
    else:
        train_loader = iidp.data.DataLoader(
            train_data, batch_size=args.train_batch_size, batch_fn=batch_func, loading_once=False,
            collate_fn=collate_fn, sampler=train_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=False)

    train_sampler = train_loader.sampler

    # get validate and translation data loaders
    args.val_batch_size = args.train_batch_size if args.train_batch_size < 32 else 32
    args.test_batch_size = args.train_batch_size if args.train_batch_size < 32 else 32
    print('[INFO] args.val_batch_size', args.val_batch_size)
    print('[INFO] args.test_batch_size', args.test_batch_size)
    val_loader = val_data.get_loader(batch_size=args.val_batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=0)

    # If args.worker > 0 , then raise RuntimeError: unable to open shared memory object
    # Reference: https://discuss.pytorch.org/t/runtimeerror-unable-to-open-shared-memory-object-depending-on-the-model/116090
    # We just leave it to 0 to avoid such error
    test_loader = test_data.get_loader(batch_size=args.test_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       pad=True,
                                       num_workers=0)

    translator = Translator(model=dp_trainer.eval_model,
                            batch_first=batch_first,
                            tokenizer=tokenizer,
                            loader=test_loader,
                            beam_size=args.beam_size,
                            max_seq_len=args.test_max_length,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            print_freq=args.print_freq,
                            reference=args.test_tgt,
                            )

    if args.is_adaptive_training:
        total_train_iters = len(train_loader.dataset) * args.epochs
        logging.info(
            f'[Adaptive] total_train_iters: {total_train_iters} | '
            f'len(train_loader.dataset): {len(train_loader.dataset)} | '
            f'epochs: {args.epochs}')
    else:
        total_train_iters = len(train_loader.dataset) // args.train_global_batch_size * args.epochs
        logging.info(
            f'total_train_iters: {total_train_iters} | '
            f'len(train_loader.dataset): {len(train_loader.dataset)} | '
            f'GBS: {args.train_global_batch_size} | epochs: {args.epochs}')

    # Create optimizer and lr scheduler
    params = dp_trainer.main_model.parameters()
    if dp_trainer.weight_sync_method == 'overlap':
        if 'grad_clip' not in opt_config.keys():
            opt_config['grad_clip'] = args.grad_clip
            optimizer = ShardGNMTAdam(params, **opt_config)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](params, **opt_config)
    logging.info(f'Using optimizer: {optimizer}')
    if args.is_adaptive_training:
        scheduler = AdaptiveWarmupMultiStepLR(
            optimizer, total_train_iters, train_loader, **scheduler_config)
    else:
        scheduler = WarmupMultiStepLR(optimizer, total_train_iters, **scheduler_config)
    logging.info(f'Using LR scheduler: {scheduler}')
    dp_trainer.prepare_weight_sync_method(optimizer, scheduler)

    save_info = {
        'model_config': model_config,
        'config': args,
        'tokenizer': tokenizer.get_state()
        }
    loss_scaling = {
        'init_scale': args.init_scale,
        'upscale_interval': args.upscale_interval
        }
    trainer_options = dict(
        model=dp_trainer.main_model,
        batch_first=batch_first,
        grad_clip=args.grad_clip,
        print_freq=args.print_freq,
        save_info=save_info,
        save_dir=args.save_dir,
        translator=translator,
        num_minibatches=args.num_minibatches,
        dp_trainer=dp_trainer,
        is_adaptive_training=args.is_adaptive_training,
        )
    # create trainer
    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load(args.resume)

    # training loop
    training_perf = []
    break_training = False
    test_bleu = None

    for epoch in dp_trainer.remaining_epochs(args.epochs):
        trainer.epoch = epoch
        logging.info(f'Starting epoch {epoch}')
        if args.distributed and train_sampler:
            train_loader.sampler.set_epoch(epoch)

        with dp_trainer.measure_epoch_time(), dp_trainer.record_epoch_data():
            train_loss, train_perf = trainer.optimize(train_loader)
        training_perf.append(train_perf)

        # evaluate on validation set
        if args.eval:
            logging.info(f'Running validation on dev set')
            val_loss, val_perf = trainer.evaluate(val_loader)

            # remember best prec@1 and save checkpoint
            if args.rank == 0:
                best_loss = min(val_loss, best_loss)
                if args.save_dir:
                    trainer.save(args.save_dir)

        if args.eval:
            utils.barrier()
            eval_fname = f'eval_epoch_{epoch}'
            eval_path = os.path.join(args.eval_dir, eval_fname)
            _, eval_stats = translator.run(
                calc_bleu=True,
                epoch=epoch,
                eval_path=eval_path,
                )
            test_bleu = eval_stats['bleu']
            if args.target_bleu and test_bleu >= args.target_bleu:
                logging.info(f'Target accuracy reached')
                break_training = True

        acc_log = []
        acc_log += [f'Summary: Epoch: {epoch}']
        acc_log += [f'Training Loss: {train_loss:.4f}']
        if args.eval:
            acc_log += [f'Validation Loss: {val_loss:.4f}']
            acc_log += [f'Test BLEU: {test_bleu:.2f}']

        perf_log = []
        perf_log += [f'Performance: Epoch: {epoch}']
        perf_log += [f'Training: {train_perf:.0f} Tok/s']
        if args.eval:
            perf_log += [f'Validation: {val_perf:.0f} Tok/s']

        if args.rank == 0:
            logging.info('\t'.join(acc_log))
            logging.info('\t'.join(perf_log))

        logging.info(f'Finished epoch {epoch}')
        if break_training:
            break

    utils.barrier()

    if args.is_adaptive_training is False:
        table = TrainingTable()
        avg_training_perf = sum(training_perf) / len(training_perf)
        table.add(utils.get_world_size(), args.train_global_batch_size, args.local_batch_size,
                  test_bleu, avg_training_perf, dp_trainer.total_epoch_time)
        if utils.get_rank() == 0:
            table.write('Training Summary', args.math)

        passed = utils.benchmark(test_bleu, args.target_bleu,
                                train_perf, args.target_perf)
        if not passed:
            sys.exit(1)


if __name__ == '__main__':
    main()
