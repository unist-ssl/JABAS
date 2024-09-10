import grpc

import trainer_to_scheduler_pb2 as t2s_pb2
import trainer_to_scheduler_pb2_grpc as t2s_pb2_grpc
import common_pb2


class TrainerRpcClient:

    def __init__(self, trainer_id, worker_id, sched_ip_addr, sched_port):
        self._trainer_id = trainer_id
        self._worker_id = worker_id
        self._sched_loc = '%s:%d' % (sched_ip_addr, sched_port)

    def init(self):
        request = t2s_pb2.InitJobRequest(trainer_id=self._trainer_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = t2s_pb2_grpc.TrainerToSchedulerStub(channel)
            stub.InitJob(request)

    def update_config(self, config, local_batch_size):
        request = t2s_pb2.UpdateConfigRequest(config=config, local_batch_size=local_batch_size)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = t2s_pb2_grpc.TrainerToSchedulerStub(channel)
            stub.UpdateConfig(request)

    def shutdown(self):
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = t2s_pb2_grpc.TrainerToSchedulerStub(channel)
            stub.ShutDown(common_pb2.Empty())
