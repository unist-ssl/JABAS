import torch
import torch.distributed as dist

from iidp.ddp_comm_hooks.default_hooks import _allreduce_sum_fut
from iidp.ddp_comm_hooks.iidp_allreduce import iidp_allreduce_hook, IIDPState


class GradientSimilarityState(IIDPState):
    def __init__(self, process_group, total_num_decoupled_workers, sub_group, grad_placeholder, interval):
        super().__init__(process_group, total_num_decoupled_workers)
        self.sub_group = sub_group
        self.grad_placeholder = grad_placeholder
        assert self.total_num_decoupled_workers % 2 == 0, \
            f"self.total_num_decoupled_workers must be power of 2, but {self.total_num_decoupled_workers}"
        self.subgroup_total_num_decoupled_workers = self.total_num_decoupled_workers / 2
        self.interval = interval
        self.step = 1 # To avoid first step because of rebuilding DDP bucket


def subgroup_allreduce_hook(
    state: GradientSimilarityState, bucket: dist._GradBucket
) -> torch.futures.Future:
    if state.step % state.interval != 0:
        fut = torch.futures.Future()
        fut.set_result(bucket.get_tensors())
        return fut
    # Detach bucket's tensor not to be affected by all-reduce in all process group (allgroup_allreduce())
    tensor = bucket.get_tensors()[0].detach().clone()
    group_to_use = state.sub_group
    future_work = _allreduce_sum_fut(group_to_use, tensor)
    def append_to_grad_placeholder(fut):
        state.grad_placeholder.append(fut.value()[0].div_(state.subgroup_total_num_decoupled_workers))
        return [fut.value()[0]]

    return future_work.then(append_to_grad_placeholder)


def similimarity_allreduce_hook(hook):
    def hook_with_allreduce(state, bucket):
        future_work = hook(state, bucket)
        def allgroup_allreduce(fut):
            iidp_allreduce_hook(state, bucket).wait()
            return bucket.get_tensors()
        return future_work.then(allgroup_allreduce)
    return hook_with_allreduce