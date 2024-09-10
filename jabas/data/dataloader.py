import torch.distributed as dist

from iidp.data.dataloader import DataLoader
from iidp.train import LOCAL_TRAINER_STATE


class AdaptiveDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, batch_fn=None, size_fn=None, loading_once=None, **kwargs):
        super().__init__(dataset, batch_size, batch_fn, loading_once, **kwargs)

        if loading_once is True: # TODO
            raise ValueError(f'Not support with Argument "loading_once" = True')

        if size_fn is None:
            raise ValueError(f'Argument "size_fn" must be configured by user, but: {size_fn}')
        self.size_fn = size_fn

    def __iter__(self):
        self.data_index = 0
        self.done = False
        iter_idx = 0
        num_yielded = 0
        if self.loading_once is True: # TODO
            raise RuntimeError(f'Not support with Argument "loading_once" = True')
        else:
            # NOTE: Since self._index_sampler.batch_size is changed to local batch size,
            # len(super().__iter__()) is also modified.
            self._index_sampler.batch_size = LOCAL_TRAINER_STATE.local_batch_size
            local_batch_data = []
            while not self.done:
                print(f'[INFO][jabas.data.AdaptiveDataLoader] rank: {dist.get_rank()} | Initial loading.. it might take time..')
                # NOTE: iidp.data.DataLoader.__iter__() is called
                for idx, batch in enumerate(super(type(self).__bases__[0], self).__iter__()):
                    if self.size_fn(batch) != self.batch_sampler.batch_size:
                        continue
                    # NOTE: index drawn from super().__iter__() is initialized when it is over
                    iter_idx += 1
                    if len(local_batch_data) < self.total_local_num_models:
                        local_batch_data.append(batch)
                    if len(local_batch_data) == self.total_local_num_models:
                        # NOTE: after yielding, self.global_batch_size and self.accum_step might be changed
                        global_batch_size_progress = self.global_batch_size
                        accum_progress = self.accum_step
                        chunked_batch = self.batch_fn(local_batch_data, self.total_local_num_models, self.loading_once)
                        yield chunked_batch
                        num_yielded += 1
                        local_batch_data = []
                        if num_yielded % (accum_progress+1) == 0:
                            self.data_index += global_batch_size_progress
                            num_yielded = 0
                    if self.data_index >= len(self.dataset):
                        self.done = True
                        break

        if self.done is False:
            raise RuntimeError(f'[ERROR][jabas.data.AdaptiveDataLoader] Flag done is not True even iterator is finished')
        self.epoch += 1

    def get_progress(self):
        return len(self.dataset) * self.epoch + self.data_index

    def update_local_state(self, batch_size, total_local_num_models, accum_step):
        if self.loading_once is True:
            self.batch_sampler.batch_size = batch_size
        else:
            self.batch_sampler.batch_size = batch_size // total_local_num_models
            self.curr_sampler_iter_len = len(self.batch_sampler)
        self.total_local_num_models = total_local_num_models
        self.accum_step = accum_step

    def update_global_state(self, global_batch_size, partition_size):
        self.global_batch_size = global_batch_size