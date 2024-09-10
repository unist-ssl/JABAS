import logging
import os
import socket
import signal
import sys
import threading

from runtime.rpc import dispatcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

from iidp.utils.json_utils import read_json


LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class Worker:
    def __init__(self, worker_id, sched_addr, sched_port, worker_port,
                 cmd, initial_url, initial_config, initial_local_batch_size,
                 checkpoint_dir, gpu_cluster_info_file, log_dir=None):
        logger = logging.getLogger('worker')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._logging_handler = ch

        signal.signal(signal.SIGINT, self._signal_handler)

        self.gpu_cluster_info = read_json(gpu_cluster_info_file)
        self.server_info = self.gpu_cluster_info[socket.gethostname()]
        num_gpus = self.server_info['number']
        self._device_id = self.server_info['type']
        self._worker_id = worker_id
        self._worker_addr = socket.gethostbyname(socket.gethostname())
        self._worker_port = worker_port
        self._worker_rpc_client = worker_client.WorkerRpcClient(
                self._device_id, self._worker_id, self._worker_addr,
                self._worker_port, sched_addr, sched_port)

        callbacks = {
            'RunJob': self._run_job_callback,
            'Reset': self._reset_callback,
            'Shutdown': self._shutdown_callback,
        }

        self._server_thread = threading.Thread(
            target=worker_server.serve,
            args=(worker_port, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()

        error = self._worker_rpc_client.register_worker(num_gpus)
        if error:
            raise RuntimeError(error)

        if not os.path.isdir(checkpoint_dir):
            # Set up a new checkpoint directory if does not already exist.
            os.mkdir(checkpoint_dir)

        if not initial_url:
            raise ValueError(f'Worker must have initial URL for distributed training, but: {initial_url}')

        if not initial_config:
            raise ValueError(f'Worker must have initial IIDP configuration (VSW, GA), but: {initial_config}')

        if not initial_local_batch_size:
            raise ValueError(f'Worker must have initial local batch size, but: {initial_local_batch_size}')

        self._dispatcher = dispatcher.Dispatcher(self._worker_rpc_client,
                                                 sched_addr,
                                                 sched_port,
                                                 cmd,
                                                 initial_url,
                                                 initial_config,
                                                 initial_local_batch_size,
                                                 checkpoint_dir,
                                                 worker_id,
                                                 self.server_info,
                                                 self.gpu_cluster_info,
                                                 log_dir)

    def _run_job_callback(self, trainer_ids, world_size: int, master_addr="", config={}, local_batch_size=0):
        # hack to prevent a job being dispatched before the dispatcher is set up
        # TODO: fix this by sending a "I'm ready" message to scheduler
        while True:
            try:
                self._dispatcher
                break
            except Exception as e:
              continue
        self._logger.debug(f'Dispatching run request from scheduler client gRPC')
        self._dispatcher.dispatch(trainer_ids, world_size, master_addr, config, local_batch_size)

    def _signal_handler(self, sig, frame):
        self._dispatcher.shutdown(killed=True)
        self._logger.removeHandler(self._logging_handler)
        self._logging_handler.close()
        sys.exit(0)

    def _reset_callback(self):
        self._dispatcher.reset()

    def _shutdown_callback(self):
        self._dispatcher.shutdown()
        self._logger.removeHandler(self._logging_handler)
        self._logging_handler.close()

    def join(self):
        self._server_thread.join()