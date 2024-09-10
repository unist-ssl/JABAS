from concurrent import futures
import time
import threading

import grpc
import logging
import os
import sys
import socket
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc
import trainer_to_scheduler_pb2 as t2s_pb2
import trainer_to_scheduler_pb2_grpc as t2s_pb2_grpc
import common_pb2

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class SchedulerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger

    def RegisterWorker(self, request, context):
        register_worker_callback = self._callbacks['RegisterWorker']
        try:
            succeed, worker_id, err = register_worker_callback(
                                        device_id=request.device_id,
                                        worker_id=request.worker_id,
                                        num_gpus=request.num_gpus,
                                        ip_addr=request.ip_addr,
                                        port=request.port)
            if succeed:
                self._logger.info(
                    'Successfully registered GPU: {device_id} in worker: '
                    '{worker_id}'.format(
                        device_id=request.device_id,
                        worker_id=worker_id))
                return w2s_pb2.RegisterWorkerResponse(success=succeed,
                                                      worker_id=worker_id)
            else:
                self._logger.error('Could not register worker: {0}'.format(err))
                return w2s_pb2.RegisterWorkerResponse(success=succeed,
                                                      error_message=err)
        except Exception as e:
            self._logger.error('Could not register worker: {0}'.format(e))
            return w2s_pb2.RegisterWorkerResponse(success=False,
                                                  error_message=e)

    def Done(self, request, context):
        done_callback = self._callbacks['Done']
        try:
            trainer_ids, succeed = done_callback(request.worker_id)
            if succeed:
                self._logger.info(
                    'Received completion notification: '
                    'Trainer IDs: {trainer_ids}, Worker ID: {worker_id}'.format(
                        trainer_ids=str(trainer_ids), worker_id=request.worker_id))
            else:
                self._logger.error('Could not process completion '
                                'notification for worker {}'.format(request.worker_id))
        except Exception as e:
            self._logger.error('Could not process completion '
                               'notification for worker {}'.format(request.worker_id))
            traceback.print_exc()

        return common_pb2.Empty()

    def Killed(self, request, context):
        killed_callback = self._callbacks['Killed']
        try:
            succeed = killed_callback(request.worker_id)
            if succeed:
                self._logger.info(
                    'Received killed notification: '
                    'Worker ID: {worker_id}'.format(worker_id=request.worker_id))
            else:
                self._logger.error('Could not process killed '
                                'notification for worker {}'.format(request.worker_id))
        except Exception as e:
            self._logger.error('Could not process completion '
                               'notification for worker {}'.format(request.worker_id))
            traceback.print_exc()

        return common_pb2.Empty()


class SchedulerTrainerRpcServer(t2s_pb2_grpc.TrainerToSchedulerServicer):
    def __init__(self, callbacks, logger, terminate_condition):
        self._callbacks = callbacks
        self._logger = logger
        self._terminate_condition = terminate_condition

    def InitJob(self, request, context):
        trainer_id = request.trainer_id
        init_job_callback = self._callbacks['InitJob']
        err = init_job_callback(trainer_id)
        if err is not None:
            self._logger.error(err)
        else:
            self._logger.info('Received job initialization request '
                              'from trainer {0}'.format(trainer_id))
        return common_pb2.Empty()

    def UpdateConfig(self, request, context):
        update_job_callback = self._callbacks['UpdateConfig']
        succeed = update_job_callback(request.config, request.local_batch_size)
        if succeed:
            self._logger.info("Updated Configuration...")
        return common_pb2.Empty()

    def ShutDown(self, request, context):
        shutdown_job_callback = self._callbacks['ShutDown']
        err = shutdown_job_callback()
        if err is not None:
            self._logger.error(err)
        else:
            self._logger.info('Received job shutdown request')
            self._terminate_condition.acquire()
            self._terminate_condition.notify()
            self._terminate_condition.release()
        return common_pb2.Empty()


def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor())
    terminate_condition = threading.Condition()
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerRpcServer(callbacks, logger), server)
    t2s_pb2_grpc.add_TrainerToSchedulerServicer_to_server(
            SchedulerTrainerRpcServer(callbacks, logger, terminate_condition), server)
    ip_address = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port('%s:%d' % (ip_address, port))
    logger.info('Starting server at {0}:{1}'.format(ip_address, port))
    server.start()

    # Wait for Trainer server to receive a shutdown RPC from scheduler.
    with terminate_condition:
        terminate_condition.wait()
    time.sleep(5)
    logger.info('Terminating server at {0}:{1}'.format(ip_address, port))
    server.stop(0)