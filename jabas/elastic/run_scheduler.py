import argparse

import scheduler

parser = argparse.ArgumentParser(description='Scheduler Runner for Elastic Training')
parser.add_argument('-c', '--config_file', type=str, required=True,
                    help="Initial configuration JSON file path")
parser.add_argument('-p', '--port', type=int, default=40000,
                    help="Scheduler's Port Number")


def main():
    args = parser.parse_args()
    # Instantiate scheduler.
    sched = scheduler.Scheduler(args.port, args.config_file)
    try:
        sched.join()
    finally:
        sched.shut_down()


if __name__=='__main__':
    main()