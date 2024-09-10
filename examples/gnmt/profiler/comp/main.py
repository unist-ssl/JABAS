import argparse

from iidp.profiler import ComputationProfiler
from comp_profiler import GNMTProfiler


parser = argparse.ArgumentParser(description='Computation Profiler')
parser.add_argument('--local-batch-size', '-lbs', default=None, type=int, required=True,
                    help='Local batch size to be preserved')
parser.add_argument('--num-models', type=int, default=1, help='Number of VSWs')
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Directory of profile data file.')


def main():
    args = parser.parse_args()

    profiler_instance = GNMTProfiler(args.local_batch_size, args.num_models)
    comp_profiler = ComputationProfiler(profiler_instance, args.profile_dir)
    comp_profiler.run()


if __name__ == '__main__':
    main()