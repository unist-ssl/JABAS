import argparse

from ddp_bucket_profiler import ResNet50Profiler
from iidp.profiler import DDPBucketProfiler


parser = argparse.ArgumentParser(description='DDP Bucket Profiler')
parser.add_argument('--profile-dir', type=str, default=None,
                    help='Directory of profile data file.')


def main():
    args = parser.parse_args()

    profiler_instance = ResNet50Profiler()
    ddp_bucket_profiler = DDPBucketProfiler(
            profiler_instance, args.profile_dir)
    ddp_bucket_profiler.run()


if __name__ == '__main__':
    main()