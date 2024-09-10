import os
import math

from iidp.utils.distributed import get_allgather_value, print_one_rank
from iidp.utils.json_utils import read_json
from iidp.utils.global_vars import MAX_MEM_PROFILE_FILE_NAME

from iidp.config.configurator import IIDPConfigurator, IIDPStaticLocalBatchSizeConfigurator

from jabas.config.config_utils import sorted_listdir, sorted_config_map_by_rank


class AdaptiveBatchingConfigurator(IIDPConfigurator):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_dir, local_config, global_server_info,
                 max_global_batch_size, is_dynamic_local_batch_size=False, gpu=None,
                 num_gpu_alloc_unit=2):
        ADAPTIVE_BATCHING_CONFIG_MSG = \
            "For adaptive batching with similarity, we divide two identical GPU groups. " \
            "To create two groups with identical IIDP configuration, one allocation unit is considered as two GPUs."
        print_one_rank(f'[INFO] {ADAPTIVE_BATCHING_CONFIG_MSG}')
        assert num_gpu_alloc_unit == 2, \
            "If number of GPU allocation unit for adaptive batching with similarity is changed, " \
            "the compatablity of determinig the rounded number of virtual workers must be considered in " \
            "AdaptiveBatchingConfigurator.solve_placement() and AutoScalingConfigurator.estimate_epoch_time()"
        super().__init__(comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                         memory_profile_dir, local_config, global_server_info,
                         max_global_batch_size, is_dynamic_local_batch_size, gpu,
                         num_gpu_alloc_unit=num_gpu_alloc_unit)

        assert is_dynamic_local_batch_size == True, \
            f"Not support static local batch size for adaptive batching"

        self.iidp_config_map_in_cluster, self.updated_iidp_config_map = {}, {}
        self._init_config_map_in_cluster(local_config)

    def _init_config_map_in_cluster(self, local_config):
        all_num_models_in_process_group = {
            rank: value for rank, value in enumerate(get_allgather_value(local_config.num_models, self.gpu))
        }
        all_accum_step_in_process_group = {
            rank: value for rank, value in enumerate(get_allgather_value(local_config.accum_step, self.gpu))
        }
        for rank, (num_models, accum_step) in enumerate(
                zip(all_num_models_in_process_group.values(), all_accum_step_in_process_group.values())):
            self.iidp_config_map_in_cluster[rank] = (num_models, accum_step)

    def _convert_cluster_config_set_to_rank_config_map(self, new_config_set):
        def _generate_config_map(config_set: list):
            def _convert_config_str_to_int(config_str):
                return int(config_str.split(':')[-1])
            config_map = {} # {rank: (num_models, accum_step)}
            for config_name in config_set:
                region_str, num_models_str, accum_step_str = config_name.split(',')
                head_rank = _convert_config_str_to_int(region_str)
                num_models = _convert_config_str_to_int(num_models_str)
                accum_step = _convert_config_str_to_int(accum_step_str)
                for i in range(self.num_gpu_alloc_unit):
                    config_map[head_rank+i] = (num_models, accum_step)
            return config_map

        new_config_map = sorted_config_map_by_rank(_generate_config_map(new_config_set))
        return new_config_map

    def update(self):
        """
        Lazy update - After IIDP trainer actually updates all of states,
        configuration map will be updated by being called update()
        """
        if len(self.updated_iidp_config_map) == 0:
            raise ValueError(f'self.updated_iidp_config_map must not be empty, '
                             f'but {self.updated_iidp_config_map} => '
                             f'solve_placement() may return empty config map')
        self.iidp_config_map_in_cluster = self.updated_iidp_config_map

    def solve_placement(self, global_batch_size, current_global_batch_size):
        new_iidp_config_map = {} # return value
        best_throughput = -1
        best_solved_iidp_config_map = {}
        best_local_batch_size = 0
        best_total_num_workers = 0
        new_global_batch_size = 0
        for local_batch_size, configurator in self.configurators.items():
            if global_batch_size > current_global_batch_size: # Increase global batch size
                total_num_workers = round(global_batch_size/local_batch_size)
                total_num_workers += (total_num_workers % 2) # Similarity constraint
                if local_batch_size*total_num_workers <= current_global_batch_size:
                    continue
            else: # Decrease global batch size
                total_num_workers = round(global_batch_size/local_batch_size)
                total_num_workers -= (total_num_workers % 2) # Similarity constraint
                if local_batch_size*total_num_workers >= current_global_batch_size:
                    continue

            if total_num_workers < self.total_num_gpus:
                continue
            throughput, _, _, cluster_config_map = configurator.solve_dynamic_programming(total_num_workers)
            solved_iidp_config_map = self._convert_cluster_config_set_to_rank_config_map(cluster_config_map)
            if solved_iidp_config_map == {}:
                continue
            if throughput > best_throughput:
                best_local_batch_size = local_batch_size
                best_total_num_workers = total_num_workers
                best_throughput = throughput
                best_solved_iidp_config_map = solved_iidp_config_map
                new_global_batch_size = best_local_batch_size * best_total_num_workers

        if best_solved_iidp_config_map == {}:
            return new_iidp_config_map, best_local_batch_size, new_global_batch_size

        try:
            for rank in range(self.total_num_gpus):
                new_num_models = best_solved_iidp_config_map[rank][0] - self.iidp_config_map_in_cluster[rank][0]
                new_accum_step = best_solved_iidp_config_map[rank][1] - self.iidp_config_map_in_cluster[rank][1]
                if new_num_models == 0 and new_accum_step == 0:
                    continue
                new_iidp_config_map[rank] = (new_num_models, new_accum_step)
        except Exception as e:
            print_one_rank(f'[{self.__class__.__name__}] solve_placement() | rank: {rank} | '
                            f'best_solved_iidp_config_map: {best_solved_iidp_config_map} | '
                            f'self.iidp_config_map_in_cluster: {self.iidp_config_map_in_cluster}', 'error')
            raise e
        if new_iidp_config_map:
            self.updated_iidp_config_map = best_solved_iidp_config_map
        return new_iidp_config_map, best_local_batch_size, new_global_batch_size


class AutoScalingConfigurator(object):
    def __init__(self, comp_profile_dir, comm_profile_dir, bucket_profile_dir,
                 memory_profile_dir, local_config, candidate_global_server_infos,
                 max_global_batch_size=-1, is_dynamic_local_batch_size=False,
                 gpu=None, num_gpu_alloc_unit=None):
        self.comp_profile_dir = comp_profile_dir
        self.comm_profile_dir = comm_profile_dir
        self.bucket_profile_dir = bucket_profile_dir

        if len(sorted_listdir(comp_profile_dir)) != len(sorted_listdir(memory_profile_dir)):
            raise ValueError(
                f'[ERROR][{self.__class__.__name__}] '
                f'Computation and memory profile data for range of local batch size '
                f'must be equal, but comp: {sorted_listdir(comp_profile_dir)} | mem: {sorted_listdir(memory_profile_dir)}')
        # Create memory profile info (type: dict)
        self.memory_profile_info = {}
        for lbs in sorted_listdir(memory_profile_dir):
            if not lbs in self.memory_profile_info.keys():
                self.memory_profile_info[lbs] = {}
            static_lbs_mem_profile_dir = os.path.join(memory_profile_dir, lbs)
            # Check the same server profile data in computation and memory profile dir
            if os.listdir(os.path.join(self.comp_profile_dir, lbs)) != os.listdir(static_lbs_mem_profile_dir):
                raise ValueError(
                    f'[ERROR] For static LBS of {lbs}, server profile data is not consistent among comp and memory profile dir!\n'
                    f'comp profile dir - {os.path.join(self.comp_profile_dir, lbs)} : {os.listdir(os.path.join(self.comp_profile_dir, lbs))}\n'
                    f'memory profile dir - {static_lbs_mem_profile_dir} : {os.listdir(static_lbs_mem_profile_dir)}')
            for server_name in os.listdir(static_lbs_mem_profile_dir):
                max_memory_profile_file = os.path.join(
                    static_lbs_mem_profile_dir, server_name, MAX_MEM_PROFILE_FILE_NAME)
                memory_profile_json_data = read_json(max_memory_profile_file)
                self.memory_profile_info[lbs][memory_profile_json_data['gpu_type']] = memory_profile_json_data['max_num_models']

        self.local_config = local_config
        if not isinstance(max_global_batch_size, int) or max_global_batch_size < 0:
            raise TypeError(
                f'Argument ```max_global_batch_size``` must be positive integer, '
                f'but {max_global_batch_size} and type: {type(max_global_batch_size)}')
        self.max_global_batch_size = max_global_batch_size
        # NOTE: candidate global server info is defined by list type in [jabas/cluster/cluster_manager.py]
        if not isinstance(candidate_global_server_infos, list):
            candidate_global_server_infos = [candidate_global_server_infos]
        self.candidate_global_server_infos = candidate_global_server_infos
        self.is_dynamic_local_batch_size = is_dynamic_local_batch_size
        self.gpu = gpu
        self.num_gpu_alloc_unit = num_gpu_alloc_unit or 2 # 2 is for similarity constraint

        self.static_lbs = local_config.lbs if self.is_dynamic_local_batch_size is False else -1
        # [Set of IIDPConfigurator for all global server info]
        # built once by prepare() at the initial phase of elastic training
        self.all_candidate_server_configurators = {}
        # [IIDPConfigurator for each global server info]
        # IIDPConfigurator => set of IIDPStaticLocalBatchSizeConfigurator
        self.configurators = {}

        self._prepared = False
        self.current_global_batch_size = 0
        self.current_local_batch_size = 0
        self._update_lock = False

    def _convert_cluster_config_set_to_rank_config_map(self, new_config_set):
        def _generate_config_map(config_set: list):
            def _convert_config_str_to_int(config_str):
                return int(config_str.split(':')[-1])
            config_map = {} # {rank: (num_models, accum_step)}
            for config_name in config_set:
                region_str, num_models_str, accum_step_str = config_name.split(',')
                head_rank = _convert_config_str_to_int(region_str)
                num_models = _convert_config_str_to_int(num_models_str)
                accum_step = _convert_config_str_to_int(accum_step_str)
                for i in range(self.num_gpu_alloc_unit):
                    config_map[head_rank+i] = (num_models, accum_step)
            return config_map

        new_config_map = sorted_config_map_by_rank(_generate_config_map(new_config_set))
        return new_config_map

    def state_dict(self):
        return self.all_candidate_server_configurators

    def prepare(self, verbose=True):
        if len(self.all_candidate_server_configurators) == 0:
            self._init_configurators(verbose)
        self._prepared = True

    def _init_configurators(self, verbose=True):
        print_one_rank(
            '==============================================================\n'
            f'[{self.__class__.__name__}] Start to initialize configurators for candidate servers\n'
            f'[{self.__class__.__name__}] Total number of candidate severs to build: {len(self.candidate_global_server_infos)}\n'
            f'[{self.__class__.__name__}] verbose: {verbose}\n'
            f'[{self.__class__.__name__}] It might take time ..\n'
            '=============================================================='
        )
        for server_id, global_server_info in enumerate(self.candidate_global_server_infos):
            total_num_gpus = global_server_info.total_num_gpus
            self.all_candidate_server_configurators[server_id] = {}
            configurators = {}
            all_server_names = []
            for server_info in global_server_info:
                all_server_names.append(server_info.name)
            # Instantiate configurator for static local batch size
            for lbs in sorted_listdir(self.comp_profile_dir):
                try:
                    local_batch_size = int(lbs)
                except:
                    print_one_rank(
                        f'[{self.__class__.__name__}] init_configurators() '
                        f'Computation profile dir structure is not suitable for local batch size: '
                        f'{sorted_listdir(self.comp_profile_dir)}', 'error')
                    exit(1)
                if self.is_dynamic_local_batch_size is False and local_batch_size != self.local_config.lbs:
                    continue
                static_lbs_comp_profile_dir = os.path.join(self.comp_profile_dir, lbs)
                # Check current local batch size can be supported by current global servers
                if not set(all_server_names).issubset(set(os.listdir(static_lbs_comp_profile_dir))):
                    if verbose is True:
                        print_one_rank(
                            f'[{self.__class__.__name__}] init_configurators() '
                            f'local_batch_size: {local_batch_size} is not supported '
                            f'by current cluster: {global_server_info} ==> skip it for IIDP configuration'
                        )
                    continue
                if self.max_global_batch_size//local_batch_size < total_num_gpus:
                    if verbose is True:
                        print_one_rank(
                            f'[{self.__class__.__name__}] init_configurators() '
                            f'local_batch_size: {local_batch_size} '
                            f'is not satisfied with current total number of GPUs: {total_num_gpus} '
                            f'==> skip it for IIDP configuration'
                        )
                    continue
                static_lbs_memory_profile_info = self.memory_profile_info[lbs]
                max_num_workers = self.max_global_batch_size//local_batch_size+1
                configurators[local_batch_size] = IIDPStaticLocalBatchSizeConfigurator(
                    static_lbs_comp_profile_dir, self.comm_profile_dir, self.bucket_profile_dir,
                    static_lbs_memory_profile_info, local_batch_size,
                    self.local_config.weight_sync_method, global_server_info, max_num_workers,
                    self.num_gpu_alloc_unit
                )
            self.all_candidate_server_configurators[server_id] = configurators

            if verbose is True:
                final_result_log_str = \
                    f'[{server_id} / {len(self.candidate_global_server_infos)}] ' \
                    f'server id: {server_id} | ' \
                    f'all_server_names: {all_server_names} | ' \
                    f'total number of GPUs: {global_server_info.total_num_gpus}'
                length = len(final_result_log_str) + 1
                print_one_rank('=' * length)
                print_one_rank(final_result_log_str)
                print_one_rank('=' * length)

        # NOTE: Check if at least one of the candidate servers can support the initial local batch size
        is_support_initial_lbs = False
        for server_id, configurators in self.all_candidate_server_configurators.items():
            if self.local_config.lbs in configurators.keys():
                is_support_initial_lbs = True
        if is_support_initial_lbs is False:
            raise ValueError(
                f'No candidate server to support such initial local batch size: {self.local_config.lbs}'
            )
        print_one_rank(
            '==============================================================\n'
            f'[{self.__class__.__name__}] Finish to initialize configurators for candidate servers\n'
            '=============================================================='
        )

    def update(self, server_id, local_batch_size, global_batch_size):
        if self._prepared is False:
            raise RuntimeError(
                f'[ERROR][{self.__class__.__name__}] update() must be called '
                f'after prepare() is called'
            )
        # Update current local & global batch size -> must be preserved for next epoch
        self.current_global_batch_size = global_batch_size
        self.current_local_batch_size = local_batch_size
        # Update configurators for each static local batch size with global resource
        self.configurators = self.all_candidate_server_configurators[server_id]
        self._update_lock = True

    def estimate_epoch_time(self, global_batch_size, iteration, remaining_num_dataset):
        if self._update_lock is False:
            raise RuntimeError(
                f'[ERROR][{self.__class__.__name__}] estimate_epoch_time() must be called '
                f'after update() is called'
            )
        assert self.num_gpu_alloc_unit == 2, \
            "If number of GPU allocation unit for adaptive batching with similarity is changed, " \
            "the compatablity of determinig the rounded number of virtual workers must be considered in " \
            "AdaptiveBatchingConfigurator.solve_placement() and AutoScalingConfigurator.estimate_epoch_time()"
        min_duration, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step \
            = math.inf, {}, global_batch_size, 0
        for local_batch_size, configurator in self.configurators.items():
            # NOTE: current LBS and GBS must be preserved for next epoch
            if global_batch_size == self.current_global_batch_size:
                if self.current_local_batch_size != local_batch_size:
                    continue
                total_num_workers = global_batch_size // self.current_local_batch_size
                if total_num_workers >= configurator.total_num_gpus and iteration > 0:
                    _, iter_time, _, cluster_config_map = configurator.solve_dynamic_programming(total_num_workers)
                    solved_iidp_config_map = self._convert_cluster_config_set_to_rank_config_map(cluster_config_map)
                    if solved_iidp_config_map == {}: # Candidate global server cannot support current global batch size
                        return min_duration, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
                    if remaining_num_dataset - global_batch_size*iteration < 0:
                        expected_step = (remaining_num_dataset // global_batch_size) + 1
                    else:
                        expected_step = iteration
                    if expected_step <= 0:
                        return min_duration, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
                    duration = iter_time * expected_step
                    return duration, solved_iidp_config_map, global_batch_size, expected_step
                else:
                    return min_duration, ret_solved_iidp_config_map, ret_expected_gbs, ret_expected_step
            else:
                if global_batch_size > self.current_global_batch_size: # Increasing global batch size
                    total_num_workers = round(global_batch_size/local_batch_size)
                    total_num_workers += (total_num_workers % 2) # Similarity constraint
                    if local_batch_size*total_num_workers < self.current_global_batch_size:
                        continue
                else:
                    total_num_workers = round(global_batch_size/local_batch_size)
                    total_num_workers -= (total_num_workers % 2) # Similarity constraint
                    if local_batch_size*total_num_workers > self.current_global_batch_size:
                        continue
                if total_num_workers >= configurator.total_num_gpus and iteration > 0:
                    _, iter_time, _, cluster_config_map = configurator.solve_dynamic_programming(total_num_workers)
                    solved_iidp_config_map = self._convert_cluster_config_set_to_rank_config_map(cluster_config_map)
                    if solved_iidp_config_map == {}:
                        continue
                    expected_gbs = total_num_workers * local_batch_size
                    if remaining_num_dataset - expected_gbs*iteration < 0:
                        expected_step = (remaining_num_dataset // expected_gbs) + 1
                    else:
                        expected_step = iteration
                    if expected_step <= 0:
                        continue
                    duration = iter_time * expected_step
                    if duration < min_duration:
                        min_duration = duration
                        ret_solved_iidp_config_map = solved_iidp_config_map
                        ret_expected_gbs = expected_gbs
                        ret_expected_step = expected_step
        return min_duration, ret_solved_iidp_config_map, \
                ret_expected_gbs, ret_expected_step