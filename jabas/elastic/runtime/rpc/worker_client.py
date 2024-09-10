import grpc
import logging
import os
import sys
import socket
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class WorkerRpcClient:
    """Worker client for sending RPC requests to a scheduler server."""

    def __init__(self, device_id, worker_id, worker_ip_addr, worker_port,
                 sched_ip_addr, sched_port):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._device_id = device_id
        self._worker_id = worker_id
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_ip_addr = sched_ip_addr
        self._sched_port = sched_port
        # TODO: Remove self._sched_ip_addr and self._sched_port?
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)

    def register_worker(self, num_gpus):
        request = w2s_pb2.RegisterWorkerRequest(
            device_id=self._device_id,
            worker_id=self._worker_id,
            num_gpus=num_gpus,
            ip_addr=self._worker_ip_addr,
            port=self._worker_port)
        with grpc.insecure_channel(self._sched_loc) as channel:
            self._logger.debug('Trying to register worker...')
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.RegisterWorker(request)
            if response.success:
                self._logger.info(
                    f'Succesfully registered worker ID (=rank): {self._worker_id} '
                    f'| server hostname: {socket.gethostname()}'
                )
                return None
            else:
                assert(response.HasField('error_message'))
                self._logger.error(f'Failed to register worker | server hostname: {socket.gethostname()}')
                return response.error_message

    def notify_scheduler(self, worker_id, killed=False):
        # Send a Done message.
        if killed:
            request = w2s_pb2.KilledRequest(worker_id=worker_id)
        else:
            request = w2s_pb2.DoneRequest(worker_id=worker_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            if killed:
                stub.Killed(request)
            else:
                stub.Done(request)
