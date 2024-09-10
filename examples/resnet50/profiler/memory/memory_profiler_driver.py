import argparse
import os

from jabas.profiler import AdaptiveStaticLocalBatchSizeMultiGPUMemoryProfiler, \
                            AdaptiveDynamicLocalBatchSizeMultiGPUMemoryProfiler


parser = argparse.ArgumentParser()
parser.add_argument('--profile-dir', type=str, default=None, required=True,
                    help='Directory of profile data file.')
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int,
                    help='Local batch size for profiling')
parser.add_argument('--min-batch-size', default=None, type=int,
                    help='Min local batch size for profiling')
parser.add_argument('--max-batch-size', default=None, type=int,
                    help='Max local batch size for profiling')


def main():
    args = parser.parse_args()

    if args.local_batch_size is None and args.min_batch_size is None:
        raise ValueError(f'One of argument --local-batch-size or --min-batch-size must be configured')
    if args.local_batch_size is not None and args.min_batch_size is not None:
        raise ValueError(f'Not support both arguments --local-batch-size and --min-batch-size')

    is_static_lbs = (args.local_batch_size is not None)
    is_dynamic_lbs = (args.min_batch_size is not None)

    if is_dynamic_lbs:
        if os.path.exists(args.profile_dir):
            raise ValueError(f'--profile-dir {args.profile_dir} must not exist')

        def search_lbs_fn(lbs):
            return lbs + args.min_batch_size

    MAX_PROFILE_ITER = 20
    data_dir = os.path.join(os.getenv('JABAS_DATA_STORAGE'), 'imagenet')
    CMD_TEMPLATE = f"""python main.py -a resnet50 \
        --dist-url tcp://localhost:32005 \
        --dist-backend nccl \
        --multiprocessing-distributed \
        --world-size 1 \
        --rank 0 \
        --num-minibatches {MAX_PROFILE_ITER} \
        --no-validate \
        --weight-sync-method recommend \
        -lbs %(local_batch_size)d \
        --num-models %(num_models)d \
        --accum-step %(accum_step)d \
        --jabas-config-file %(jabas_config_file)s \
        {data_dir}"""

    if is_static_lbs:
        mem_profiler = AdaptiveStaticLocalBatchSizeMultiGPUMemoryProfiler(
            args.profile_dir, CMD_TEMPLATE, args.local_batch_size)
    elif is_dynamic_lbs:
        mem_profiler = AdaptiveDynamicLocalBatchSizeMultiGPUMemoryProfiler(
            args.profile_dir, CMD_TEMPLATE, args.min_batch_size,
            search_lbs_fn, args.max_batch_size)
    mem_profiler.run()


if __name__ == '__main__':
    main()