from jabas.elastic.runtime.rpc import scheduler_server, scheduler_client
import threading
import time
import json
import copy


class WorkerInfo:
    def __init__(self, worker_id, device_name, max_num_gpus, sched_rpc_client,
                 trainer_ids=None):
        self.worker_id = worker_id
        self.device_name = device_name
        self.max_num_gpus = max_num_gpus
        self.rpc_client = sched_rpc_client
        self.trainer_ids = trainer_ids if trainer_ids is not None else []

        start_rank = worker_id * max_num_gpus
        self.ranks = list(range(start_rank, start_rank+max_num_gpus))


class Timer:
    def __init__(self, start_time):
        self.start_time = start_time
        self.elapsed_time = 0

    def update(self, measured_time):
        self.elapsed_time = measured_time - self.start_time


class Scheduler:
    def __init__(self, scheduler_port, init_file):
        # Synchronization primitives to ensure thread-safe updates of
        # scheduler metadata.
        self._scheduler_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._scheduler_lock)
        # List to Maintain Global Information
        """{ Worker ID : WorkerInfo} """
        self._worker_list = {}
        self._rank_to_worker_id_map = {}
        self._worker_changed = False
        self._need_to_reschedule = False
        self._init = True
        self._initialized = False
        self._updated = False
        self._finished = False
        with open(init_file, "r") as json_file:
            self._init_workers = json.load(json_file, object_pairs_hook=lambda pairs: {int(k): v for k, v in pairs})
        self._updated_config = {}
        self._updated_local_batch_size = 0
        self.initialize_overhead_timer = None
        self.initialize_overhead = -1

        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'InitJob': self._init_job_callback,
            'UpdateConfig': self._update_config_callback,
            'Done': self._done_callback,
            'Killed': self._killed_callback,
            'ShutDown': self.shut_down
        }

        self._reschedule_thread = \
            threading.Thread(target=self._reschedule_thread)
        self._reschedule_thread.daemon = True
        self._reschedule_thread.start()

        self._server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(scheduler_port, callbacks))
        self._server_thread.daemon = True
        self._server_thread.start()

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, device_id: str, worker_id: int,
                                  num_gpus: int, ip_addr: str, port: int):
        if worker_id in self._worker_list:
            return (False, -1, "Agent already running.")
        if num_gpus <= 0:
            return (False, -1, "No Device to work with.")
        # Share a single RPC client for each GPU on the worker.
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)

        with self._scheduler_lock:
            self._worker_changed = True
            self._worker_list[worker_id] = WorkerInfo(worker_id, device_id, num_gpus, rpc_client)
            for trainer_id in self._worker_list[worker_id].ranks:
                self._rank_to_worker_id_map[trainer_id] = worker_id

            if worker_id in self._init_workers:
                if not (num_gpus >= self._init_workers[worker_id] and self._init_workers[worker_id] >= 0):
                    self._shut_down()
                if self.initialize_overhead_timer is None:
                    self.initialize_overhead_timer = Timer(time.time())

            if not self._initialized:
                initialize_finished = all(id in self._worker_list for id in self._init_workers)
                if initialize_finished:
                    self._init = True
                    self._initialized = True
                    self._need_to_reschedule = True
                    self._scheduler_cv.notifyAll()
        return (True, worker_id, "")

    def _init_job_callback(self, trainer_id: int):
        with self._scheduler_lock:
            if trainer_id not in self._rank_to_worker_id_map:
                return "Unknown Trainer ID : {}".format(trainer_id)
            if self.initialize_overhead < 0:
                self.initialize_overhead_timer.update(time.time())
                self.initialize_overhead = self.initialize_overhead_timer.elapsed_time
            self._updated = False
        return None

    def _update_config_callback(self, config, local_batch_size):
        with self._scheduler_lock:
            if len(config) == 0 and not self._worker_changed:
                return False
            if self._updated:
                return False
            self._worker_changed = False
            sorted_config = dict(sorted(config.items()))
            self._updated_config.update(sorted_config)
            self._updated_local_batch_size = local_batch_size
            self._need_to_reschedule = True
            self._init = False
            self._updated = True
            self._scheduler_cv.notifyAll()
        return True

    def _done_callback(self, worker_id: int):
        # Every job in worker is done
        with self._scheduler_lock:
            if worker_id not in self._worker_list:
                return None, False
            trainer_ids = copy.deepcopy(self._worker_list[worker_id].trainer_ids)
            self._worker_list[worker_id].trainer_ids.clear()
            return trainer_ids, True

    def _killed_callback(self, worker_id: int):
        # Worker is killed
        with self._scheduler_lock:
            if worker_id not in self._worker_list:
                return False
            self._worker_changed = True

            del self._worker_list[worker_id]
            if worker_id in self._updated_config:
                del self._updated_config[worker_id]
            if len(self._worker_list) == 0:
                self._initialized = False
            return True

    """
    ======================================================================
       Public-facing scheduler methods.
    ======================================================================
    """

    def _run_trainer(self, worker_id, trainer_ids, world_size, master_addr="", trainer_config_map={}):
        rpc_client = self._worker_list[worker_id].rpc_client
        local_batch_size = self._updated_local_batch_size
        rpc_client.run_job(trainer_ids, world_size, master_addr, trainer_config_map, local_batch_size)

    def _reset_workers(self):
        # rpc_client and trainer IDs are reset in self._worker_list
        for worker_id in self._worker_list.keys():
            self._worker_list[worker_id].rpc_client.reset()
            self._worker_list[worker_id].trainer_ids.clear()

    def shut_down(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            if self._finished:
                return
            for worker in self._worker_list.values():
                worker.rpc_client.shutdown()
            self._finished = True

    def _shut_down(self):
        if self._finished:
            return
        for worker in self._worker_list.values():
            worker.rpc_client.shutdown()
        self._finished = True
        exit(-1)

    def join(self):
        self._server_thread.join()

    """
    ======================================================================
       Helper methods to get and mutate state needed for scheduling.
    ======================================================================
    """

    def _reschedule_thread(self):
        """Computes the VSWs/Resource Rescheduling asynchronously."""
        while True:
            # Check whether rescheduling needs to be re-computed.
            self._scheduler_cv.acquire()
            while not self._need_to_reschedule:
                self._scheduler_cv.wait()
            self._scheduler_cv.release()
            # Reschedule
            with self._scheduler_lock:
                self._reschedule()
                self._updated_config.clear()
                self._need_to_reschedule = False
                self._init = False

    def _reschedule(self):
        new_worker_list = {}
        if self._init:
            new_worker_list.update(self._init_workers)
            rank = 0
            world_size = sum(new_worker_list.values())
            for worker_id in sorted(self._worker_list.keys()):
                if worker_id in new_worker_list:
                    new_trainer_ids = list(set(range(rank, rank + new_worker_list[worker_id])))
                    self._worker_list[worker_id].trainer_ids = new_trainer_ids
                    self._run_trainer(worker_id, new_trainer_ids, world_size)
                    rank += new_worker_list[worker_id]
        else:
            for rank, config_str in self._updated_config.items():
                worker_id = self._rank_to_worker_id_map[rank]
                if worker_id not in new_worker_list.keys():
                    new_worker_list[worker_id] = {}
                new_worker_list[worker_id].update({rank: config_str})
            self._reset_workers()
            rank = 0
            world_size = len(self._updated_config.keys())
            master_addr = ""
            for worker_id in sorted(self._worker_list.keys()):
                if worker_id in new_worker_list:
                    trainer_ids = list(set(range(rank, rank+len(new_worker_list[worker_id]))))
                    new_trainer_config_map = {}
                    for idx, (_, config_str) in enumerate(new_worker_list[worker_id].items()):
                        new_rank = trainer_ids[idx]
                        new_trainer_config_map[new_rank] = config_str
                    self._worker_list[worker_id].trainer_ids = trainer_ids
                    if 0 in trainer_ids: # If rank 0 is included, it is master worker (server)
                        master_addr = self._worker_list[worker_id].rpc_client.addr
                    if not master_addr:
                        raise ValueError(f"[ERROR][jabas/elastic/scheduler.py] master_addr must be configured, but {master_addr}")
                    self._run_trainer(worker_id, trainer_ids, world_size, master_addr, new_trainer_config_map)
                    rank += len(new_worker_list[worker_id])

        if not (rank == world_size):
            print(f'[ERROR][jabas/elastic/scheduler.py] _reschedule() - rank: {rank} != world_size: {world_size}!')
            self._shut_down()