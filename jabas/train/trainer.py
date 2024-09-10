from contextlib import contextmanager
from collections import defaultdict
import math
import os
import time
import datetime

import torch
import torch.distributed as dist

import threading
import copy
import gc

import iidp
from iidp.utils.json_utils import read_json
from iidp.utils.global_vars import CHECKPOINT_FILE_NAME
from iidp.train.trainer import IIDPTrainer
from iidp.config.configurator import IIDPConfig

import jabas

from jabas.elastic.runtime.rpc import trainer_client as trainer_client

from jabas.config.configurator import AdaptiveBatchingConfigurator, AutoScalingConfigurator
from jabas.cluster.cluster_manager import JABASClusterManager

from jabas.config.model.global_batch_size.gaussian_process import GaussianProcessRegressionModel
from jabas.config.model.global_batch_size.exponential_smoothing import ExponentialSmoothing
from jabas.config.model.global_batch_size.ensemble_method import EnsembleMethod

from jabas.utils.cost_utils import estimate_cost
from jabas.utils.server import get_resource_info_dict
from jabas.config.config_utils import check_user_config_is_valid
from jabas.profiler.memory.profile_utils import get_mem_profile_data_summary
from jabas.utils.timer import Timer


class JABASTrainer(IIDPTrainer):
    def __init__(self, gpu, local_batch_size, num_models, accum_step,
                 weight_sync_method='recommend', config_params=None,
                 checkpoint_dir_for_elastic=None, restart_timer_for_elastic=None):
        super().__init__(gpu, local_batch_size, num_models, accum_step, weight_sync_method)
        self.data_loader = None
        self.is_elastic_training = False
        self.is_resource_reallocated = False
        self._checkpoint_dir_for_elastic = checkpoint_dir_for_elastic
        self._checkpoint_path = None
        if self._checkpoint_dir_for_elastic is not None:
            self.is_elastic_training = True
            self._trainer_id = int(os.environ['JABAS_TRAINER_ID'])
            self._local_rank = int(os.environ['JABAS_LOCAL_RANK'])
            self._worker_id = int(os.environ['JABAS_WORKER_ID'])
            self._sched_addr = os.environ['JABAS_SCHED_ADDR']
            self._sched_port = int(os.environ['JABAS_SCHED_PORT'])
            self._rpc_client = trainer_client.TrainerRpcClient(
                    self._trainer_id, self._worker_id, self._sched_addr, self._sched_port)
            if not os.path.isdir(self._checkpoint_dir_for_elastic):
                raise ValueError(f'self._checkpoint_dir_for_elastic must be directory: {self._checkpoint_dir_for_elastic}')
            self._checkpoint_path = os.path.join(self._checkpoint_dir_for_elastic, 'checkpoint.pth')

            self.is_resource_reallocated = (os.path.exists(self._checkpoint_path) is True)
            # NOTE: Include the below elapsed time
            # 1) re-initialize and train components setup (model, optimizer, data loader, etc)
            # 2) save & load checkpoint overhead
            self.reallocation_overhead = -1
            self.restart_timer_for_elastic = restart_timer_for_elastic

        self.is_load_checkpoint_by_user = False
        self.config_params = config_params
        self._check_user_config_is_valid()
        self._init_config_params()
        # For IIDP dynamic configuration
        self.prev_max_num_models = self.num_models

        self.global_batch_size_trajectory = []
        if self.is_elastic_training is True:
            self._prepare_global_batch_size_prediction()

        self.total_epoch_cost = 0

    def prepare_stream_parallel(self, model, criterion, **kwargs):
        if not kwargs.get('gradient_as_bucket_view'):
            kwargs['gradient_as_bucket_view'] = True
            self.trainer_print('gradient_as_bucket_view must be True')
        super().prepare_stream_parallel(model, criterion, **kwargs)

    def prepare_weight_sync_method(self, optimizer, scheduler=None, param_groups_func=None):
        self._set_local_optimizers(optimizer, param_groups_func)
        self._set_local_schedulers(scheduler)
        self._prepare_gradient_based_metric()
        if self.prepared_for_ddp:
            if self.weight_sync_method == 'overlap':
                self._prepare_overlap_optimizer_with_ddp()
            if self.weight_sync_method == 'sequential':
                self._register_comm_hook()
        else:
            raise RuntimeError("[ERROR] Without DDP, AdaptiveIIDPTrainer cannot work")

    def prepare_adaptive_data_loader(self, data_loader):
        if not isinstance(data_loader, jabas.data.AdaptiveDataLoader):
            raise ValueError(f'Only support jabas.data.AdaptiveDataLoader, but {type(data_loader)}')
        self.data_loader = data_loader

    def save(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path): # checkpoint_path is directory
            checkpoint_dir = checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE_NAME)
        else: # checkpoint_path is file
            checkpoint_dir = os.path.dirname(checkpoint_path)

        trainer_state_dict = self._state_dict_for_ckpt_based_restart()

        # NOTE: To confirm the consistency of the last saved configuration
        trainer_state_dict['global_batch_size'] = self.global_batch_size
        trainer_state_dict['local_batch_size'] = self.local_batch_size
        trainer_state_dict['iidp_config_map_in_cluster'] = self.adaptive_batching_configurator.iidp_config_map_in_cluster

        self.trainer_print(f'Save trainer state to {checkpoint_path}')
        torch.save(trainer_state_dict, checkpoint_path)

        if self.is_elastic_training is True:
            self.batch_size_model.save(checkpoint_dir)

    def load(self, checkpoint_path, restrict_saved_config=False):
        if not os.path.exists(checkpoint_path):
            raise FileExistsError(f'Checkpoint path: {checkpoint_path} does not exist')
        if not os.path.isfile(checkpoint_path): # checkpoint_path is directory
            checkpoint_dir = checkpoint_path
            checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE_NAME)
        else: # checkpoint_path is file
            checkpoint_dir = os.path.dirname(checkpoint_path)

        loc = 'cuda:{}'.format(self.gpu) if type(self.gpu) == int else self.gpu
        state_dict = torch.load(checkpoint_path, map_location=loc)
        self._load_state_dict_for_ckpt_based_restart(state_dict)

        if self.is_elastic_training is True:
            self.batch_size_model.load(checkpoint_dir)

        if restrict_saved_config:
            # [CHECK 1] Confirm the resume resource setup is equal to the previous saved one
            self.trainer_print(f'[load_state_dict] The saved IIDP config on cluster setup: {state_dict["iidp_config_map_in_cluster"]}')
            all_ranks = list(state_dict['iidp_config_map_in_cluster'].keys())
            if len(all_ranks) != dist.get_world_size():
                raise ValueError(
                    f'[load_state_dict] Current number of GPUs: {dist.get_world_size()} '
                    f'is not equal to the saved number of GPUs: {len(all_ranks)}')
            global_batch_size_in_current_cluster = 0
            for _, (num_models, accum_step) in state_dict['iidp_config_map_in_cluster'].items():
                global_batch_size_in_current_cluster += int(state_dict['local_batch_size'] * (num_models*(accum_step+1)))
            if global_batch_size_in_current_cluster != state_dict['global_batch_size']:
                raise ValueError(
                    f'[load_state_dict] The saved global batch size: {state_dict["global_batch_size"]} '
                    f'is not equal to the global batch size on the cluster : {global_batch_size_in_current_cluster}')

            # [CHECK 2] Confirm the resume IIDP configuration is equal to the previous saved one
            assert self.global_batch_size == state_dict['global_batch_size'], \
                f"self.global_batch_size: {self.global_batch_size} | state_dict['global_batch_size']: {state_dict['global_batch_size']}"
            assert self.local_batch_size == state_dict['local_batch_size'], \
                f"self.local_batch_size: {self.local_batch_size} | state_dict['local_batch_size']: {state_dict['local_batch_size']}"
            assert self.num_models == state_dict['iidp_config_map_in_cluster'][dist.get_rank()][0], \
                f"self.num_models: {self.num_models} | saved num_models: {state_dict['iidp_config_map_in_cluster'][dist.get_rank()][0]}"
            assert self.accum_step == state_dict['iidp_config_map_in_cluster'][dist.get_rank()][1], \
                f"self.accum_step: {self.accum_step} | saved accum_step: {state_dict['iidp_config_map_in_cluster'][dist.get_rank()][1]}"

        # NOTE: USAGE - See [_load_checkpoint_for_elastic_training()]
        self.is_load_checkpoint_by_user = True

    def compute(self, data):
        if self.config_params["enable_adjust"] and self.local_accum_step == -1:
            self._adjust_adaptive_lr()
        super().compute(data)

    def step(self):
        # NOTE: As IIDP has overlapping backward pass and optimizer.step(),
        # scaled LR must be adopted before forward pass in compute() method
        if super().step() is False:
            return False

        if self.config_params["enable_adjust"]:
            self._adjust_adaptive_lr(intialize=True)
        if self.config_params["metric"] == 'similarity' and \
                self.similarity_state.step % self.similarity_state.interval == 0:
            self._compute_cosine_similarity()
        if self.config_params["enable_adjust"] and \
                self.similarity_state.step % self.similarity_state.interval == 0:
            self._adjust_adaptive_global_batch_size()

            self.trainer_print(f"step {self.similarity_state.step} - New local batch size {self.local_batch_size}")
            self.trainer_print(f"step {self.similarity_state.step} - New global batch size {self.global_batch_size}\n==============================")

        if self.config_params["metric"] == 'similarity':
            self.similarity_state.step += 1

        return True

    @contextmanager
    def measure_epoch_time(self):
        try:
            start_time = time.time()
            yield
        finally:
            self.elapsed_time = int(time.time() - start_time)
            if self.is_resource_reallocated:
                self.elapsed_time += int(self.reallocation_overhead)
                self.trainer_print(f'Epoch time: {self.elapsed_time} ' \
                                   f'(include reallocation time: {self.reallocation_overhead:.2f} sec)')
            else:
                self.trainer_print(f'Epoch time: {self.elapsed_time}')

    def remaining_epochs(self, final_epochs):
        try:
            self._prepare_jabas_training()
            self.epoch_iterator.final_epochs = final_epochs
            for epoch in self.epoch_iterator.__iter__():
                self.global_batch_size_trajectory.append([])
                if self.is_elastic_training is True:
                    self._current_resource_info_parser()
                yield epoch
                is_last_epoch = (epoch == len(self.epoch_iterator)-1)
                if self.is_elastic_training is True:
                    if self._trainer_id == 0 and not is_last_epoch:
                        # NOTE: If auto-scaling is requested,
                        # the below code after the method at rank 0 do not reach
                        self._prepare_joint_adaptive_training_by_forecasting()
                    if is_last_epoch:
                        # NOTE: If not last epoch, it is called at _prepare_joint_adaptive_training_by_forecasting()
                        self._measure_cost()
                        self.total_epoch_time += self.elapsed_time
                dist.barrier()
                self.is_resource_reallocated = False
        finally:
            self._print_final_results()

        dist.barrier()
        if self.is_elastic_training is True:
            # NOTE: _local_rank is defined only if is_elastic_training = True
            if self._local_rank == 0:
                self._rpc_client.shutdown()

    def _current_resource_info_parser(self):
        resource_info_dict = get_resource_info_dict(self.cluster_manager.global_server_info)
        self.trainer_print(f'Current resource info: {resource_info_dict}')

    def _measure_cost(self):
        total_cost_per_epoch = 0
        for server_info in self.cluster_manager.global_server_info:
            total_cost_per_epoch += estimate_cost(
                    server_info.resource_info.tfplos,
                    server_info.resource_info.num_gpus_in_server,
                    self.elapsed_time / 3600 # convert to sec to hour
                )
        self.trainer_print(f'Epoch cost: {total_cost_per_epoch:.2f}')
        self.total_epoch_cost += total_cost_per_epoch

    def _print_final_results(self):
        self.trainer_print('======================================')
        if self.is_elastic_training is True:
            self.trainer_print(f'Total epoch time (sec): {int(self.total_epoch_time)}')
            self.trainer_print(f'Total epoch cost (dollar): {self.total_epoch_cost:.2f}')
            self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=int(self.total_epoch_time))}')
        else:
            self.trainer_print(f'Total epoch time (sec): {int(self.total_epoch_time)}')
            self.trainer_print(f'Total epoch time: {datetime.timedelta(seconds=int(self.total_epoch_time))}')
        self.trainer_print('======================================')

    def _check_user_config_is_valid(self):
        if self.epoch == 0 and self.is_resource_reallocated is False:
            check_user_config_is_valid(self.config_params)

    def _init_config_params(self):
        """Document: jabas/docs/CONFIG.md"""
        if self.config_params is not None:
            if os.path.isfile(self.config_params):
                self.config_params = read_json(self.config_params)
            else:
                raise ValueError(f'Adaptive config param file: {self.config_params} must exist')
            self.config_params=defaultdict(lambda: None, self.config_params)
            if self.is_elastic_training is True:
                if self.config_params["batch_size_lower_bound"] is None:
                    raise ValueError(
                        f'If is_elastic_training = True, config_params["batch_size_lower_bound"] must be configured')
                else:
                    if self.is_resource_reallocated is False:
                        if self.global_batch_size < self.config_params["batch_size_lower_bound"]:
                            raise ValueError(
                                f'Within elastic training, initial global batch size: {self.global_batch_size} '
                                f'must be < batch_size_lower_bound: {self.config_params["batch_size_lower_bound"]}'
                            )
                        self.config_params["original_batch_size"] = self.global_batch_size
                if self.config_params["available_servers"] is None:
                    raise ValueError(
                        f'If is_elastic_training = True, config_params["available_servers"] must be configured')
            else:
                self.config_params["original_batch_size"] = self.global_batch_size
                if self.config_params["batch_size_lower_bound"] is None:
                    self.config_params["batch_size_lower_bound"] = self.config_params["original_batch_size"]
                    self.trainer_print(f'batch_size_lower_bound is configured by initial global batch size: {self.global_batch_size}')
            if self.config_params["batch_size_lower_bound"] is not None and self.config_params["batch_size_upper_bound"] is not None:
                assert self.config_params["batch_size_upper_bound"]>=self.config_params["batch_size_lower_bound"]
            self.config_params["global_lr_modifier"] = 1.0
            if self.config_params["enable_adjust"] is None:
                self.config_params["enable_adjust"] = True
            else:
                if isinstance(self.config_params["enable_adjust"], str): # handle to parse from json file
                    self.config_params["enable_adjust"] = bool(self.config_params["enable_adjust"] == "True")
            self.config_params["enable_adjust_lbs"] = True
            if self.config_params["metric"] is None:
                self.config_params["metric"] = 'similarity'
            if self.config_params["batch_size_adjust_rate"] is None:
                self.config_params["batch_size_adjust_rate"] = 0.1 # 10%
            if self.config_params["batch_size_adjust_interval"] is None:
                self.config_params["batch_size_adjust_interval"] = 100
            else:
                # handle to parse from json file
                if isinstance(self.config_params["batch_size_adjust_interval"], str):
                    self.config_params["batch_size_adjust_interval"] = int(self.config_params["batch_size_adjust_interval"])
            if self.config_params["available_servers"] is None:
                self.config_params["available_servers"] = []
        else:
            raise ValueError(f'JABAS configuration parameters must be configured')

        self.trainer_print(f'JABAS configuration parameters: {self.config_params}')

    def _prepare_overlap_optimizer_with_ddp(self):
        self._create_stream_for_optimizer()
        self._check_overlap_with_ddp()
        for i, (local_model, state, hook) in enumerate(zip(self.local_models, self.states, self.hooks)):
            if i == 0:
                hook = self._create_optimizer_hook(hook)
            local_model.register_comm_hook(state=state, hook=hook)

    def _register_comm_hook(self):
        for _, (local_model, state, hook) in enumerate(zip(self.local_models, self.states, self.hooks)):
            local_model.register_comm_hook(state=state, hook=hook)

    def _prepare_jabas_training(self):
        # NOTE: Resource configuration is set up by the assumption of checkpoint-based restart
        if self.config_params["enable_adjust"] is True:
            self.available_server_name_list = self.config_params["available_servers"]
            self.cluster_manager = JABASClusterManager(
                    self.config_params["gpu_cluster_info"], self.available_server_name_list,
                    self.config_params["homo_servers"], self.config_params["resource_alloc_unit"], self.gpu)
        self._prepare_configuration_solver()

        if self.local_models == [] or self.local_optimizers == [] or self.data_loader is None:
            raise ValueError(
                f'Before calling prepare_adaptive_training(), model, optimizer and data loader must be configured')
        if self.is_elastic_training is True:
            self._prepare_checkpoint_based_restart()
            # NOTE: Pre-build configuratos for all candidate global resource
            self.auto_scale_configurator.prepare()

    """
    ======================================================================
        Configuration Solver
    ======================================================================
    """

    def _print_initial_config(self):
        if self.epoch == 0 and self.is_resource_reallocated is False:
            cluster_config_str = ''
            for server_info in self.cluster_manager.global_server_info:
                cluster_config_str += (server_info.__repr__() + '\n')
            self.trainer_print(
                f'\n====================== Initial configuration ======================\n'
                f'-------------------------- Cluster --------------------------------\n'
                f'{cluster_config_str}'
                f'-------------------------------------------------------------------\n'
                f'GBS: {self.global_batch_size} | LBS: {self.local_batch_size} | '
                f'IIDP config: {self.adaptive_batching_configurator.iidp_config_map_in_cluster}\n'
                f'===================================================================='
            )
            self.trainer_print(
                f'\n========== Memory Profile Data Summary ==========\n'
                f'   LBS\t|\tGPU\t|\tMax number of VSWs\n'
                f'---------------------------------------------------\n'
                f'{get_mem_profile_data_summary(self.config_params["memory_profile_dir"])}'
                f'==================================================='
            )

    def _prepare_configuration_solver(self):
        # NOTE
        # 1) Profile data dir must be placed on each local server even GPU type is same among another servers
        # 2) Porfile data on all of local servers must be placed on every servers (e.g, NFS)
        self.local_config = IIDPConfig(self.local_batch_size, self.num_models, self.accum_step, self.weight_sync_method)
        if self.config_params["enable_adjust"] is True:
            self.adaptive_batching_configurator = AdaptiveBatchingConfigurator(
                self.config_params["comp_profile_dir"],
                self.config_params["comm_profile_dir"],
                self.config_params["bucket_profile_dir"],
                self.config_params["memory_profile_dir"],
                self.local_config,
                self.cluster_manager.global_server_info,
                self.config_params["batch_size_upper_bound"],
                self.config_params["enable_adjust_lbs"],
                self.gpu
            )
            self._print_initial_config()
            if self.is_elastic_training is True:
                self.auto_scale_configurator = AutoScalingConfigurator(
                    self.config_params["comp_profile_dir"],
                    self.config_params["comm_profile_dir"],
                    self.config_params["bucket_profile_dir"],
                    self.config_params["memory_profile_dir"],
                    self.local_config,
                    self.cluster_manager.candidate_server_infos,
                    self.config_params["batch_size_upper_bound"],
                    self.config_params["enable_adjust_lbs"],
                    self.gpu
                )

    """
    ======================================================================
        Adaptive Batching Manager
    ======================================================================
    """

    def _prepare_gradient_based_metric(self):
        if self.config_params["metric"] == 'similarity':
            self._prepare_cosine_similarity()
        else: # TODO: suuport various metrics - e.g, GNS, Norm
            raise ValueError(f'Not support other gradient-based metric except similarity: {self.config_params["metric"]}')

    def _build_subgroup(self):
        # Original similarity group-making [SimiGrad in NeurIPS '21]
        # Advantage: simple and robust for various (vsw, ga) configurations
        assert torch.distributed.get_world_size() % 2 == 0
        for i in range(torch.distributed.get_world_size()):
            self.sub_groups_idx[i%2].append(i)

    def _prepare_cosine_similarity(self):
        num_sub_groups = 2 # [SimiGrad in NeurIPS '21] build two all-reduce sub-groups
        self.sub_groups_idx = [[] for _ in range(num_sub_groups)]
        self.sub_groups = []

        self._build_subgroup()
        assert len(self.sub_groups_idx) == 2
        for idx in self.sub_groups_idx:
            self.sub_groups.append(torch.distributed.new_group(idx))
        self.first_subgroup_src_rank, self.second_subgroup_src_rank = self.sub_groups_idx[0][0], self.sub_groups_idx[1][0]
        # To compare gradients of the representative rank in each sub-group => used in _compute_cosine_similarity()
        self.sub_groups.append(torch.distributed.new_group([self.first_subgroup_src_rank, self.second_subgroup_src_rank]))

        self.grad_placeholders = [[] for _ in range(num_sub_groups)]
        self.cos_placeholder = torch.rand(1).to(self.gpu)

        self._prepare_similarity_allreduce_hooks()

    def _prepare_similarity_allreduce_hooks(self):
        self.states, self.hooks = [], []
        self.similarity_state = jabas.ddp_comm_hooks.GradientSimilarityState(
            dist.group.WORLD, self.total_num_decoupled_workers,
            self.sub_groups[dist.get_rank()%2], self.grad_placeholders[dist.get_rank()%2],
            self.config_params["batch_size_adjust_interval"]
        )
        subgroup_allreduce_hook = jabas.ddp_comm_hooks.subgroup_allreduce_hook
        main_hook = jabas.ddp_comm_hooks.similimarity_allreduce_hook(subgroup_allreduce_hook)
        dummy_hook = iidp.ddp_comm_hooks.dummy_hook
        for i in range(self.num_models):
            if i == 0:
                self.states.append(self.similarity_state)
                self.hooks.append(main_hook)
            else:
                self.states.append(None)
                self.hooks.append(dummy_hook)

    def _compute_cosine_similarity(self):
        if dist.get_rank() == self.first_subgroup_src_rank or dist.get_rank() == self.second_subgroup_src_rank:
            self.allgather_grad_placeholders = [
                torch.cat([torch.zeros_like(grad) for grad in self.similarity_state.grad_placeholder]) for _ in range(2)
            ]
            grad_placeholder = torch.cat([grad for grad in self.similarity_state.grad_placeholder])
            dist.all_gather(self.allgather_grad_placeholders, grad_placeholder, group=self.sub_groups[-1])
            if dist.get_rank() == self.first_subgroup_src_rank:
                self.cos_placeholder = torch.nn.functional.cosine_similarity(self.allgather_grad_placeholders[0], self.allgather_grad_placeholders[1], dim=0)
        dist.broadcast(self.cos_placeholder, self.first_subgroup_src_rank)
        self.similarity_state.grad_placeholder = []
        # NOTE: Memory deallocation when number of VSWs changes by _change_local_models_state()
        if dist.get_rank() == self.first_subgroup_src_rank or dist.get_rank() == self.second_subgroup_src_rank:
            del self.allgather_grad_placeholders

    def _change_local_models_state(self, adjust_num_models_diff):
        if adjust_num_models_diff > 0:
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
            torch.cuda.synchronize()

            for _ in range(adjust_num_models_diff):
                # 1) stream
                self.model_streams.append(torch.cuda.Stream())
                # 2) model
                copied_model = copy.deepcopy(self.main_model.module)
                self.original_local_models.append(copied_model)
                # 3) optimizer
                # For .zero_grad(), optimizer should be added
                # TODO: remove optimizer except main model
                cls = type(self.main_optimizer.__class__.__name__, (self.main_optimizer.__class__,), dict(self.main_optimizer.__dict__))
                if self.param_groups_func:
                    params = self.param_groups_func(copied_model)
                else:
                    params = copied_model.parameters()
                args = self._get_required_args_value(self.main_optimizer)
                copied_optimizer = cls(params, lr=self.main_optimizer.defaults['lr'], *args)
                copied_optimizer.load_state_dict(self.main_optimizer.state_dict())
                copied_optimizer.zero_grad()
                self.local_optimizers.append(copied_optimizer)
            find_unused_parameters = self.main_model.find_unused_parameters
            gradient_as_bucket_view = self.main_model.gradient_as_bucket_view
            for idx in range(self.prev_num_models, self.num_models):
                with torch.cuda.stream(self.model_streams[idx]):
                    local_ddp_module = torch.nn.parallel.IndependentIdenticalDataParallel(
                        self.original_local_models[idx], device_ids=[self.gpu], output_device=[self.gpu],
                        find_unused_parameters=find_unused_parameters,
                        gradient_as_bucket_view=gradient_as_bucket_view,
                        model_index=idx, num_local_models=self.num_models,
                        total_num_models=self.total_num_decoupled_workers,
                        sync_buffer_barrier=self._sync_buffer_barrier)
                    if self.main_model._has_rebuilt_buckets:
                        local_ddp_module.reducer.initialize_buckets(self.main_model.bucket_indices)
                        local_ddp_module._has_rebuilt_buckets = True
                    self.local_models.append(local_ddp_module)
            assert (len(self.local_models) == self.num_models) and (len(self.local_optimizers) == self.num_models)
            for i in range(self.num_models):
                self.local_models[i].reconfigure(self.num_models, self.total_num_decoupled_workers, self._sync_buffer_barrier)
            # Synchornize previous models
            for i in range(self.prev_num_models, self.num_models):
                with torch.cuda.stream(self.model_streams[i]):
                    for src_param, dst_param in \
                            zip(self.main_model.parameters(), self.local_models[i].parameters()):
                        dst_param.data.copy_(src_param.data)

            # hook - total num models
            assert self.total_num_decoupled_workers % 2 == 0
            dummy_hook = iidp.ddp_comm_hooks.dummy_hook
            for i in range(self.prev_num_models, self.num_models):
                self.states.append(None)
                self.hooks.append(dummy_hook)
                self.local_models[i].register_comm_hook(state=None, hook=dummy_hook)
            self.states[0].total_num_decoupled_workers = self.total_num_decoupled_workers
            self.states[0].subgroup_total_num_decoupled_workers = self.total_num_decoupled_workers / 2
        else:
            # Remove unused streams, models and optimizers
            for _ in range(self.num_models, self.prev_num_models):
                # NOTE: Moving models to CPU tensor and removing it enables GPU memory to be decreased
                # reference: https://discuss.pytorch.org/t/deleting-tensors-in-context-save-for-backward/122917/11
                self.local_models[-1].zero_grad(set_to_none=True)
                self.local_models[-1].cpu()
                self.original_local_models[-1].zero_grad(set_to_none=True)
                self.original_local_models[-1].cpu()
                self.local_optimizers[-1].zero_grad(set_to_none=True)
                del self.local_models[-1]
                del self.original_local_models[-1]
                del self.local_optimizers[-1]
                del self.model_streams[-1]
                del self.states[-1]
                del self.hooks[-1]
            assert (len(self.local_models) == self.num_models) and (len(self.local_optimizers) == self.num_models)
            for i in range(self.num_models):
                self.local_models[i].reconfigure(self.num_models, self.total_num_decoupled_workers, self._sync_buffer_barrier)
            if adjust_num_models_diff < 0:
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
            # hook - total num models
            assert self.total_num_decoupled_workers % 2 == 0
            self.states[0].total_num_decoupled_workers = self.total_num_decoupled_workers
            self.states[0].subgroup_total_num_decoupled_workers = self.total_num_decoupled_workers / 2
        if self.is_accum_mode:
            for i in range(self.num_models):
                with torch.cuda.stream(self.model_streams[i]):
                    self.local_optimizers[i].zero_grad()

    def _change_local_trainer_state(self, new_num_models, new_accum_step):
        """This method is called by only selective rank (GPU)"""
        self.prev_num_models = self.num_models
        if new_num_models != 0:
            self.num_models = self.prev_num_models + new_num_models
            assert self.num_models > 0, f"self.num_models must be > 0"
            # Used in seq_parallel_compute() for being block different number of VSWs on inter-node
            self.sync_accum_barrier = threading.Barrier(self.num_models)
            # It is used for _sync_params() in torch/nn/parallel/distributed.py
            self._sync_buffer_barrier = [None, None]
            if self.num_models > 1:
                self._sync_buffer_barrier = [threading.Barrier(self.num_models) for _ in range(2)]

        if new_accum_step != 0:
            self.accum_step = self.accum_step + new_accum_step

        # Data loading
        self.batch_size_per_gpu = self.local_batch_size * self.num_models
        self.data_loader.update_local_state(self.batch_size_per_gpu, self.num_models, self.accum_step)

    def _change_global_trainer_state(self, solved_iidp_config_map):
        self._get_total_num_decoupled_workers()
        assert self.global_batch_size == self.local_batch_size * self.total_num_decoupled_workers, \
            f"GBS: {self.global_batch_size} | LBS: {self.local_batch_size} | " \
            f"total num models: {self.total_num_decoupled_workers} | " \
            f"rank: {dist.get_rank()} - num_models: {self.num_models} | accum_step: {self.accum_step} | " \
            f"self.adaptive_batching_configurator.iidp_config_map_in_cluster  {self.adaptive_batching_configurator.iidp_config_map_in_cluster} | " \
            f"solved_iidp_config_map: {solved_iidp_config_map} " \
            f"=> If solved_iidp_config_map is different among rank, please check config JSON file"

        self._get_all_accum_step_in_process_group()
        self.max_accum_step = max(self.all_accum_step_in_process_group)
        self.is_accum_mode = True if self.max_accum_step > 0 else False

        self._get_all_partition_size_in_process_group()
        self.data_loader.update_global_state(
                self.global_batch_size, self.all_partition_size_in_process_group)

    def _change_configuration_for_iidp(self, solved_iidp_config_map):
        if len(solved_iidp_config_map) == 0:
            # NOTE: Even though solved_iidp_config_map is empty, local batch size can be changed.
            self.batch_size_per_gpu = self.local_batch_size * self.num_models
            self.data_loader.update_local_state(self.batch_size_per_gpu, self.num_models, self.accum_step)
            self._set_trainer_state()
            dist.barrier()
            return
        # == step 1) Change local trainer state ==
        # 1-1) numer of local models, 1-2) accum step, 1-3) batch size per GPU
        rank = dist.get_rank()
        if rank in solved_iidp_config_map:
            new_num_models, new_accum_step = solved_iidp_config_map[rank]
            self._change_local_trainer_state(new_num_models, new_accum_step)
        else:
            new_num_models, new_accum_step = 0, 0
            self._change_local_trainer_state(new_num_models, new_accum_step)
        dist.barrier()
        # == step 2) Change global trainer state ==
        # 2-1) total number of models in process group
        # 2-2) all accum step in process group -> determine is_accum_mode
        # 2-3) data loader state
        self._change_global_trainer_state(solved_iidp_config_map)
        self._set_trainer_state()
        # == step 3) Change local VSW state ==
        # 3-1) Change (create / remove) streams, models and optimizers
        # 3-2) Change communication hook state of local models
        # NOTE [IMPORTANT] Even new number of models is zero,
        # 3-2) must be done in _change_local_models_state()
        self._change_local_models_state(new_num_models)
        if len(solved_iidp_config_map) > 0:
            self.adaptive_batching_configurator.update()
        dist.barrier()

    def _change_batch_size_for_iidp(self, new_global_batch_size_by_simigrad):
        solved_iidp_config_map = {} # return
        if self.config_params["batch_size_upper_bound"] is not None:
            new_global_batch_size_by_simigrad = min(new_global_batch_size_by_simigrad, self.config_params["batch_size_upper_bound"])
        if self.config_params["batch_size_lower_bound"] is not None:
            new_global_batch_size_by_simigrad = max(new_global_batch_size_by_simigrad, self.config_params["batch_size_lower_bound"])

        if self.global_batch_size == new_global_batch_size_by_simigrad:
            return solved_iidp_config_map

        solved_iidp_config_map, new_local_batch_size, new_global_batch_size = \
            self.adaptive_batching_configurator.solve_placement(
                new_global_batch_size_by_simigrad,
                self.global_batch_size
            )
        if new_local_batch_size == 0 and new_global_batch_size == 0:
            self.trainer_print(f'Candidate global batch size by SimiGrad = {new_global_batch_size_by_simigrad}, '
                               f'but no virtual worker placement solution by DP', 'warning')
            return solved_iidp_config_map
        if new_global_batch_size//new_local_batch_size < dist.get_world_size():
            self.trainer_print(f'Candidate global batch size by SimiGrad = {new_global_batch_size_by_simigrad}, '
                               f'but cannot support on current numer of GPUs: {dist.get_world_size()}', 'warning')
            return solved_iidp_config_map

        # == Change local, global batch size == #
        self.local_batch_size = new_local_batch_size
        self.global_batch_size = new_global_batch_size
        assert self.global_batch_size % self.local_batch_size == 0, \
            f"New global batch size {self.global_batch_size} must be preserved local batch size: {self.local_batch_size}"
        return solved_iidp_config_map

    def _scale_lr(self, new_global_batch_size):
        if self.config_params["metric"] == 'similarity':
            if self.is_elastic_training is True:
                initial_batch_size = self.config_params["batch_size_lower_bound"]
            else:
                # NOTE: For some reasons, initial batch size is smaller than batch_size_lower_bound
                # Then, learning rate for batch_size_lower_bound cannot work well with smaller initial batch size
                initial_batch_size = max(
                    self.config_params["original_batch_size"], self.config_params["batch_size_lower_bound"])
            # Square-root scaling
            new_ratio = math.sqrt(new_global_batch_size / initial_batch_size)
            self.config_params["global_lr_modifier"] = new_ratio

    def _adjust_adaptive_lr(self, intialize=False):
        if self.config_params["metric"] == 'similarity':
            if not intialize:
                for param_group in self.main_optimizer.param_groups:
                    param_group['lr']*=self.config_params["global_lr_modifier"]
            else:
                for param_group in self.main_optimizer.param_groups:
                    param_group['lr']/=self.config_params["global_lr_modifier"]

    def _adjust_adaptive_global_batch_size(self):
        if self.config_params["metric"] == 'similarity':
            self.trainer_print(f"step {self.similarity_state.step} - Current cos similiarity {self.cos_placeholder} for global batch size {self.global_batch_size} ")
            # [SimiGrad] Algorithm 1 - 2) If Φ < γ, target batch size B = 1.1B, else, B = 0.9B
            # [IIDP] - target batch size with constraint to preserve local batch size
            if self.cos_placeholder < self.config_params["similarity_target"]:
                new_global_batch_size_by_simigrad = self.global_batch_size * (1 + self.config_params["batch_size_adjust_rate"])
                solved_iidp_config_map = self._change_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                self.trainer_print(f'step {self.similarity_state.step} - Current similarity < target ==> increase global batch size')
            elif self.cos_placeholder > self.config_params["similarity_target"] and self.global_batch_size > 1:
                new_global_batch_size_by_simigrad = self.global_batch_size * (1 - self.config_params["batch_size_adjust_rate"])
                solved_iidp_config_map = self._change_batch_size_for_iidp(new_global_batch_size_by_simigrad)
                self.trainer_print(f'step {self.similarity_state.step} - Current similarity > target  ==> decrease global batch size')
            self._change_configuration_for_iidp(solved_iidp_config_map)

            self._scale_lr(self.global_batch_size)
        self.global_batch_size_trajectory[self.epoch].append([self.similarity_state.step, self.global_batch_size])

    """
    ======================================================================
        Auto-Scaling Manager
    ======================================================================
    """

    def _prepare_checkpoint_based_restart(self):
        if self.is_elastic_training is False:
            raise ValueError(
                f'_prepare_checkpoint_based_restart() must be called if elf.is_elastic_training is True'
            )
        if self.is_resource_reallocated:
            self.trainer_print(f'Load checkpoint: {self._checkpoint_path}')
            self._load_checkpoint_for_elastic_training()

        # If self.reallocation_overhead > 0 after loading checkpoint,
        # it indicates the overhead has already been measured.
        if self.reallocation_overhead < 0:
            if dist.get_rank() == 0:
                self._save_checkpoint_for_elastic_training(is_overhead_profile=True)
                self._load_checkpoint_for_elastic_training(is_overhead_profile=True)
            dist.barrier()
            self.restart_timer_for_elastic.update(time.time())
            self.reallocation_overhead = self.restart_timer_for_elastic.elapsed_time
            self.trainer_print(f'Reallocation overhead = {self.reallocation_overhead:.2f} sec')

        self._rpc_client.init()

    def _state_dict_for_ckpt_based_restart(self):
        trainer_state_dict = {}
        if self.local_schedulers:
            scheduler_state = self.main_scheduler.state_dict()
            # Scheduler for adaptive training may need data loader's get_progress()
            if scheduler_state.get('data_loader'):
                scheduler_state.pop('data_loader')
        else:
            scheduler_state = None
        trainer_state_dict.update({
            'epoch': self.epoch,
            'total_epoch_time': self.total_epoch_time,
            'total_epoch_cost': self.total_epoch_cost,
            'step': self.sync_step,
            'gbs_trajectory': self.global_batch_size_trajectory,
            'model' : self.main_model.module.state_dict(),
            'optimizer'  : self.main_optimizer.state_dict(),
            'scheduler'  : scheduler_state,
            'data': self.data_loader.state_dict()
        })

        if self.config_params["metric"] == 'similarity':
            trainer_state_dict['similarity_step'] = self.similarity_state.step
            trainer_state_dict['global_lr_modifier'] = self.config_params["global_lr_modifier"]
        if self.is_elastic_training is True:
            trainer_state_dict['initial_global_batch_size'] = self.config_params["original_batch_size"]
            trainer_state_dict['reallocation_overhead'] = self.reallocation_overhead
            trainer_state_dict['auto_scale_configurator'] = self.auto_scale_configurator.state_dict()

        return trainer_state_dict

    def _load_state_dict_for_ckpt_based_restart(self, state_dict):
        for local_model in self.local_models:
            local_model.module.load_state_dict(state_dict['model'])
        for local_optimizer in self.local_optimizers:
            local_optimizer.load_state_dict(state_dict['optimizer'])
        for local_scheduler in self.local_schedulers:
            local_scheduler.load_state_dict(state_dict['scheduler'])
        self.data_loader.load_state_dict(state_dict['data'])
        self.epoch = state_dict['epoch']
        self.total_epoch_time = state_dict['total_epoch_time']
        self.total_epoch_cost = state_dict['total_epoch_cost']
        self.sync_step = state_dict['step']
        self.global_batch_size_trajectory = state_dict['gbs_trajectory']
        if self.config_params["metric"] == 'similarity':
            self.similarity_state.step = state_dict['similarity_step']
            self.config_params["global_lr_modifier"] = state_dict['global_lr_modifier']
        if self.is_elastic_training is True:
            if 'initial_global_batch_size' in state_dict.keys():
                self.config_params["original_batch_size"] = state_dict['initial_global_batch_size']
            self.reallocation_overhead = state_dict['reallocation_overhead']
            self.auto_scale_configurator.all_candidate_server_configurators = state_dict['auto_scale_configurator']

    def _save_checkpoint_for_elastic_training(self, is_overhead_profile=False):
        trainer_state_dict = self._state_dict_for_ckpt_based_restart()

        if is_overhead_profile:
            ckpt_dir_for_profile = 'profile_ckpt_overhead_dir'
            os.makedirs(ckpt_dir_for_profile, exist_ok=True)
            save_ckpt_path = os.path.join(ckpt_dir_for_profile, 'checkpoint.pth')
            epoch = -1 # NOTE: epoch idx starts from -1, refer to [EpochIterator]
            trainer_state_dict['epoch'] = epoch
        else:
            if not os.path.exists(self._checkpoint_dir_for_elastic):
                os.makedirs(self._checkpoint_dir_for_elastic)
                self.trainer_print(f'Make a checkpoint dir: {self._checkpoint_dir_for_elastic}')
            save_ckpt_path = self._checkpoint_path

        self.trainer_print(f'Save checkpoint path: {save_ckpt_path}')

        torch.save(trainer_state_dict, save_ckpt_path)

        if self.is_elastic_training is True:
            self.batch_size_model.save(self._checkpoint_dir_for_elastic)

    def _load_checkpoint_for_elastic_training(self, is_overhead_profile=False):
        if is_overhead_profile:
            ckpt_dir_for_profile = 'profile_ckpt_overhead_dir'
            os.makedirs(ckpt_dir_for_profile, exist_ok=True)
            load_ckpt_path = os.path.join(ckpt_dir_for_profile, 'checkpoint.pth')
        else:
            load_ckpt_path = self._checkpoint_path
        self.trainer_print(f'Load checkpoint path: {load_ckpt_path}')

        loc = 'cuda:{}'.format(self.gpu) if type(self.gpu) == int else self.gpu
        checkpoint = torch.load(load_ckpt_path, map_location=loc)
        if self.is_load_checkpoint_by_user and self.sync_step > checkpoint['step']: # NOTE: latest version check with load()
            self.trainer_print(
                f'Skip to load checkpoint path: {load_ckpt_path} '
                f'because load() with the lastest version is called')
            self.is_load_checkpoint_by_user = False
            return
        self._load_state_dict_for_ckpt_based_restart(checkpoint)

        if self.is_elastic_training is True:
            self.batch_size_model.load(self._checkpoint_dir_for_elastic)

        self.trainer_print(f'Loaded epoch: {checkpoint["epoch"]+1} | iterations: {self.sync_step}')
        if is_overhead_profile:
            os.system(f'rm -rf {ckpt_dir_for_profile}')

    def _prepare_joint_adaptive_training_by_forecasting(self):
        with Timer() as forecast_timer:
            predicted_gbs_trajectory = self._predict_global_batch_size()
            gbs_trajectory = [self.global_batch_size] + predicted_gbs_trajectory
            best_server_info, best_iidp_config_map, expected_gbs_trajectory \
                = self._estimate_efficient_resource(gbs_trajectory)

        # NOTE: self.elapsed_time is used to measure cost and total execution time,
        # so forecasting time is added to it
        self.elapsed_time += forecast_timer.elapsed
        self._measure_cost()
        self.total_epoch_time += self.elapsed_time

        assert len(expected_gbs_trajectory) > 0, f'len(expected_gbs_trajectory) == 0 - {expected_gbs_trajectory}'
        if best_server_info is not None and len(best_iidp_config_map) > 0 and \
                    best_server_info != self.cluster_manager.global_server_info:
            self.trainer_print('********************** Resource auto-scaling !!! **********************')
            self._request_resource_scaling(best_iidp_config_map)

    def _estimate_efficient_resource(self, gbs_trajectory):
        best_server_info = None
        best_iidp_config_map = {}
        best_expected_gbs_trajectory = gbs_trajectory
        default_step = self.config_params["batch_size_adjust_interval"]
        min_epoch_time = math.inf

        for server_id, candidate_server_info in enumerate(self.cluster_manager.candidate_server_infos):
            self.auto_scale_configurator.update(server_id, self.local_batch_size, self.global_batch_size)
            epoch_duration = 0
            init_config_map_next_epoch = {}
            remaining_num_dataset = len(self.data_loader.dataset)
            expected_gbs_trajectory = []
            for gbs_idx, gbs in enumerate(gbs_trajectory):
                step = default_step
                if gbs_idx == 0: # current global batch size
                    step = default_step - (self.sync_step % default_step)
                # With last GBS, (iteration * GBS) must process all remaining dataset
                if gbs_idx == len(gbs_trajectory)-1:
                    step = (remaining_num_dataset // gbs) + 1

                time, iidp_config_map, expected_gbs, expected_step \
                    = self.auto_scale_configurator.estimate_epoch_time(gbs, step, remaining_num_dataset)
                if time == math.inf:
                    break
                if gbs_idx == 0: # current global batch size
                    init_config_map_next_epoch = iidp_config_map
                if expected_step <= 0 and time != math.inf:
                    continue

                remaining_num_dataset -= expected_gbs*expected_step
                expected_gbs_trajectory.append(expected_gbs)
                epoch_duration += time
                if epoch_duration == math.inf:
                    break

            if (epoch_duration > 0 and epoch_duration <= min_epoch_time):
                min_epoch_time = epoch_duration
                best_server_info = candidate_server_info
                best_iidp_config_map = init_config_map_next_epoch
                best_expected_gbs_trajectory = expected_gbs_trajectory

        return best_server_info, best_iidp_config_map, best_expected_gbs_trajectory

    def _request_resource_scaling(self, rank_to_config_map):
        """NOTE: [Important] This function is called by only one rank"""
        def convert_config_map_to_proto(config_map):
            # NOTE: Message type is defined in [jabas/elastic/runtime/protobuf/trainer_to_scheuder.proto]
            config_map_proto = {}
            for rank, (num_models, accum_step) in config_map.items():
                config_map_proto[rank] = f"{num_models},{accum_step}"
            return config_map_proto

        if self.data_loader.done is False:
            raise AssertionError('Resource re-allocation must be at the end of epoch')

        self._save_checkpoint_for_elastic_training()
        # NOTE: Only one rank communicates with agent to update configuration
        config_map_proto = convert_config_map_to_proto(rank_to_config_map)
        self._rpc_client.update_config(config_map_proto, self.local_batch_size)
        # NOTE: As update_config() requests asynchronously, one rank should stop here
        while True:
            pass

    """
    ======================================================================
        GBS Forecaster
    ======================================================================
    """

    def _prepare_global_batch_size_prediction(self):
        models = [GaussianProcessRegressionModel(), ExponentialSmoothing()]
        rates = [0.5, 0.5]
        self.batch_size_model = EnsembleMethod(models, rates)

    def _predict_global_batch_size(self):
        def train_batch_size_model():
            x_train_list, y_train_list = [], []
            for step, gbs in self.global_batch_size_trajectory[self.epoch]:
                x_train_list.append(step)
                y_train_list.append(gbs)
            try:
                self.batch_size_model.train(x_train_list, y_train_list)
            except Exception as e:
                self.trainer_print(
                    f'[train_batch_size_model()] x_train_list: {x_train_list} | '
                    f'y_train_list: {y_train_list} | '
                    f'self.epoch: {self.epoch} | '
                    f'global_batch_size_trajectory at epoch: {self.global_batch_size_trajectory[self.epoch]}\n'
                    f'total global_batch_size_trajectory: {self.global_batch_size_trajectory}', 'error')
                raise e
        # Train
        if len(self.global_batch_size_trajectory[self.epoch]) > 0:
            train_batch_size_model()
        # Prepare around next steps for prediction
        x_pred_list, iidp_gbs_trajectory = [], []
        default_step = self.config_params["batch_size_adjust_interval"]
        num_dataset = len(self.data_loader.dataset)
        total_steps_at_next_epoch = num_dataset // self.global_batch_size
        number_of_adjust_batch_size_at_next_epoch = total_steps_at_next_epoch // default_step
        around_steps_at_next_epoch = []
        for i in range(1, number_of_adjust_batch_size_at_next_epoch+1):
            around_steps_at_next_epoch.append(self.sync_step + default_step*i)

        for step in around_steps_at_next_epoch:
            x_pred_list.append(step)
        # Predict
        if len(x_pred_list) > 0:
            try:
                y_pred_mean = self.batch_size_model.evaluate(x_pred_list)
            except Exception as e:
                self.trainer_print(
                    f'_predict_global_batch_size() - x_pred_list: {x_pred_list} | '
                    f'self.epoch: {self.epoch} | '
                    f'global_batch_size_trajectory at epoch: {self.global_batch_size_trajectory[self.epoch]}\n'
                    f'total global_batch_size_trajectory: {self.global_batch_size_trajectory}', 'error')
                raise e
            predicted_gbs_trajectory = y_pred_mean.ravel()
            iidp_gbs_trajectory = list(predicted_gbs_trajectory)
        return iidp_gbs_trajectory