import grpc
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
import common_pb2


class SchedulerRpcClient:
    """Scheduler client for sending RPC requests to a worker server."""

    def __init__(self, server_ip_addr, port):
        self._addr = server_ip_addr
        self._port = port
        self._server_loc = '%s:%d' % (server_ip_addr, port)

    @property
    def addr(self):
        return self._addr

    @property
    def port(self):
        return self._port

    def run_job(self, trainer_ids, world_size, master_addr="", config={}, local_batch_size=0):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.RunJobRequest(trainer_ids=trainer_ids,
                                            world_size=world_size,
                                            master_addr=master_addr,
                                            config=config,
                                            local_batch_size=local_batch_size)
            response = stub.RunJob(request)

    def reset(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            response = stub.Reset(common_pb2.Empty())

    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            stub.Shutdown(common_pb2.Empty())
