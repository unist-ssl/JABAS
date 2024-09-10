import argparse
import os
import time


from iidp.profiler import ComputationProfilerDriver


parser = argparse.ArgumentParser(description='Driver for Computation Profiler')
parser.add_argument('--mem-profile-dir', type=str, default=None, required=True,
                    help='Directory of profile data file.')
parser.add_argument('--profile-dir', type=str, default=None, required=True,
                    help='Directory of profile data file.')
parser.add_argument('--gpu-reuse-pause-time', '-p', type=int, default=300,
                    help='GPU reuse pause time to accurately profile computation')


def main():
    args = parser.parse_args()
    if not os.path.exists(args.mem_profile_dir):
        raise ValueError(f'--mem-profile-dir {args.mem_profile_dir} must exist')
    os.makedirs(args.profile_dir)

    if args.gpu_reuse_pause_time > 0:
        print(f'[INFO] Pause GPU for {args.gpu_reuse_pause_time} sec ..')
        time.sleep(args.gpu_reuse_pause_time)

    CMD_TEMPLATE = f"""CUDA_VISIBLE_DEVICES=%(gpu_id)d python profiler/comp/main.py \
        --profile-dir {args.profile_dir} \
        -lbs %(local_batch_size)d \
        --num-models %(num_models)d"""
    comp_profiler_driver = ComputationProfilerDriver(
            args.mem_profile_dir, CMD_TEMPLATE, args.gpu_reuse_pause_time)
    comp_profiler_driver.run()


if __name__ == '__main__':
    main()
