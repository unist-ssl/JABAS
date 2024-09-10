import copy
from multiprocessing.pool import ThreadPool
import logging
import os
import subprocess
import sys
import threading
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import _utils

MAX_CPUS_PER_GPU = 8
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
ITERATOR_LOG_FORMAT = '[{asctime}] [{event}] [{status}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOG_FILE_NAME = 'convergence_log.txt'


class Dispatcher:
    def __init__(self, worker_rpc_client, sched_addr, sched_port,
                 cmd, initial_url, initial_config, initial_local_batch_size,
                 checkpoint_dir, worker_id, worker_info, gpu_cluster_info, log_dir=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._thread_pool = ThreadPool()
        self._worker_rpc_client = worker_rpc_client
        self._sched_addr = sched_addr
        self._sched_port = sched_port
        self._cmd = cmd
        self._checkpoint_dir = checkpoint_dir
        self._resetted = False
        self._killed = False
        # Job management
        self._job_assignments = dict()
        self._commands = dict()
        # Lock
        self._lock = threading.Lock()
        # Worker data
        self._worker_id = worker_id
        self.initial_master_url = initial_url
        self.master_port = initial_url.split(':')[-1]
        self.worker_info = worker_info
        self.gpu_cluster_info = gpu_cluster_info
        self.num_gpus_in_server = worker_info['number']
        self._gpu_ids = list(range(self.num_gpus_in_server))
        self.initial_iidp_config = initial_config # (num_models, accum_step)
        self.initial_local_batch_size = initial_local_batch_size

        self._log_dir = log_dir

    def _construct_command(self, gpu_id, trainer_id, world_size, master_addr="", config="", local_batch_size=0):
        master_url = f'tcp://{master_addr}:{self.master_port}' if master_addr else self.initial_master_url
        num_models, accum_step = map(int, config.split(",")) if config else self.initial_iidp_config
        local_batch_size = local_batch_size if local_batch_size > 0 else self.initial_local_batch_size
        checkpoint_dir = os.path.join(self._checkpoint_dir)

        command = '%s --gpu %d' % (self._cmd, gpu_id)
        command = '%s --elastic-checkpoint-dir %s' % (command, checkpoint_dir)
        command = '%s --rank %d' % (command, trainer_id)
        command = '%s --world-size %d' % (command, world_size)
        command = '%s --is-elastic-training' % (command)
        command = '%s --dist-url %s' % (command, master_url)
        command = '%s --num-models %d' % (command, num_models)
        command = '%s --accum-step %d' % (command, accum_step)
        command = '%s --local-batch-size %d' % (command, local_batch_size)

        self._logger.info("command: {}".format(command))

        return command

    def _kill_trainers(self, trainer_id=None):
        with self._lock:
            if trainer_id is not None:
                self._logger.debug('Killing Trainer {0}...'.format(trainer_id))
            else:
                self._logger.debug('Killing all Trainer!')
            if trainer_id is not None:
                if trainer_id in self._job_assignments:
                    del self._job_assignments[trainer_id]
                if trainer_id not in self._commands:
                    return
                command = self._commands[trainer_id]
                del self._commands[trainer_id]
                _utils.kill_pid_for_job(command)
            else:
                for trainer_id in self._job_assignments.keys():
                    if trainer_id not in self._commands:
                        continue
                    command = self._commands[trainer_id]
                    _utils.kill_pid_for_job(command)
            self._logger.debug('Finished killing Trainer(s)')

    def launch(self, trainer_id, local_rank, command):
        self._logger.info('Trainer {} launched...'.format(trainer_id))

        with self._lock:
            self._commands[trainer_id] = command

        # Try to dispatch trainer.
        try:
            env = copy.deepcopy(os.environ)
            env['JABAS_TRAINER_ID'] = str(trainer_id)
            env['JABAS_LOCAL_RANK'] = str(local_rank)
            env['JABAS_WORKER_ID'] = str(self._worker_id)
            env['JABAS_SCHED_ADDR'] = self._sched_addr
            env['JABAS_SCHED_PORT'] = str(self._sched_port)

            proc = subprocess.Popen(command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        shell=True)
            if trainer_id == 0 and self._log_dir is not None:
                log_file = os.path.join(self._log_dir, LOG_FILE_NAME)
                with open(log_file, 'a') as f:
                    for line in proc.stdout:
                        print(line.decode('utf-8').replace('\n',''))
                        f.write(line.decode('utf-8'))
            else:
                for line in proc.stdout:
                    print(line.decode('utf-8').replace('\n',''))

        except subprocess.CalledProcessError as e:
            error_message = ('Trainer {trainer_id} (worker {worker_id} failed!').format(
                                     trainer_id=trainer_id, worker_id=self._worker_id)
            self._logger.error(error_message)
            traceback.print_exc()
            if e.stdout is not None:
                self._logger.debug('Trainer {trainer_id} (worker {worker_id}, '
                                   'stdout:\n{output}'.format(
                                       trainer_id=trainer_id, worker_id=self._worker_id,
                                       output=e.stdout))
            if e.stderr is not None:
                self._logger.debug('Trainer {trainer_id} (worker {worker_id}, '
                                   'stderr:\n{output}'.format(
                                       trainer_id=trainer_id, worker_id=self._worker_id,
                                       output=e.stderr))
            self._kill_trainers(trainer_id=trainer_id)
        except Exception as e:
            self._logger.error('Dispatcher failed to launch Trainer {trainer_id} '
                               '(worker {worker_id})!'.format(
                                   trainer_id=trainer_id, worker_id=self._worker_id))
            traceback.print_exc()
            self._kill_trainers(trainer_id=trainer_id)

        with self._lock:
            if not self._resetted:
                if trainer_id in self._commands:
                    del self._commands[trainer_id]
                if trainer_id in self._job_assignments:
                    del self._job_assignments[trainer_id]

        if not self._resetted:
            self._logger.info('Trainer {} finished...'.format(trainer_id))
        return 0

    def _dispatch_helper(self, trainer_ids, world_size, master_addr, config, local_batch_size):
        if len(trainer_ids) == 0 or len(trainer_ids) > self.num_gpus_in_server:
            return
        with self._lock:
            for i, trainer_id in enumerate(trainer_ids):
                self._job_assignments[trainer_id] = (self._gpu_ids[i], i)
                self._logger.debug('Trainer {} dispatched...'.format(trainer_id))

        success = True
        commands = []
        for trainer_id in trainer_ids:
            try:
                gpu_id, _ = self._job_assignments[trainer_id]
                if config:
                    trainer_config = config[trainer_id]
                else:
                    trainer_config = ""
                command = \
                    self._construct_command(
                        gpu_id, trainer_id, world_size, master_addr,
                        trainer_config, local_batch_size)
                commands.append(command)
            except Exception as e:
                self._logger.error('Failed to construct command '
                                   'for Trainer {0}!'.format(trainer_id))
                traceback.print_exc()
                success = False
                break

        sync_results = []
        if success:
            # Launch the jobs.
            for trainer_id, command in zip(trainer_ids, commands):
                _, local_rank = self._job_assignments[trainer_id]

                sync_results.append(self._thread_pool.apply_async(
                        self.launch,
                        (trainer_id, local_rank, command)))

        for sync_ in sync_results:
            sync_.get()
        # Cleanup and notify the scheduler.
        if not self._resetted:
            self._worker_rpc_client.notify_scheduler(self._worker_id)
        else:
            self._resetted = False
        return

    def dispatch(self, trainer_ids, world_size, master_addr, config, local_batch_size):
        self._thread_pool.apply_async(self._dispatch_helper,
                                      (trainer_ids, world_size, master_addr, config, local_batch_size ))

    def reset(self):
        self._resetted = True
        self._logger.debug('Resetting dispatcher...')
        self._kill_trainers()
        self._job_assignments.clear()
        self._commands.clear()
        self._thread_pool = ThreadPool()
        self._logger.debug('Finished resetting dispatcher')

    def shutdown(self, killed=False):
        if self._killed:
            return
        self._logger.debug('Shutting down dispatcher...')
        self._kill_trainers()
        self._thread_pool.terminate()
        self._logger.debug('Finished shutting down dispatcher')
        if killed:
            self._worker_rpc_client.notify_scheduler(self._worker_id,
                                                     killed=killed)
        self._killed = True
