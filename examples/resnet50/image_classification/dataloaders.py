# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial

DATA_BACKEND_CHOICES = ["pytorch", "syntetic"]


def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


# Original code is at image_classification/mixup.py
def mixup(alpha, data, target):
    with torch.no_grad():
        bs = data.size(0)
        c = np.random.beta(alpha, alpha)

        perm = torch.randperm(bs).cuda()

        md = c * data + (1 - c) * data[perm, :]
        mt = c * target + (1 - c) * target[perm, :]
        return md, mt


def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, start_epoch, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = start_epoch
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot
        )

    def __len__(self):
        return len(self.dataloader)


def get_pytorch_train_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    start_epoch=0,
    workers=5,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    traindir = os.path.join(data_path, "train")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        ),
    )

    # if torch.distributed.is_initialized():
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # else:
    #     train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        collate_fn=partial(fast_collate, memory_format),
        drop_last=True,
    )

    return (
        PrefetchedWrapper(train_loader, start_epoch, num_classes, one_hot),
        len(train_loader),
    )


def get_iidp_train_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    workers=4,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
    distributed=False,
    mixup_alpha=0.0,
    is_adaptive_training=False
):
    traindir = os.path.join(data_path, "train")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    mean = (
        torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        .cuda()
        .view(1, 3, 1, 1)
    )
    std = (
        torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        .cuda()
        .view(1, 3, 1, 1)
    )

    def prefetch_func(batch, num_chunks, loading_once):
        if loading_once is True:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda()
            targets = targets.cuda()
            inputs = inputs.float().sub_(mean).div_(std)
            if one_hot:
                targets = expand(num_classes, torch.float, targets)
            if mixup_alpha != 0.0:
                inputs, targets = mixup(mixup_alpha, inputs, targets)
            chunked_inputs = torch.tensor_split(inputs, num_chunks)
            chunked_targets = torch.tensor_split(targets, num_chunks)
            parallel_local_data = []
            for chunked_input, chunked_target in zip(chunked_inputs, chunked_targets):
                parallel_local_data.append([chunked_input, chunked_target])
            return parallel_local_data
        else:
            parallel_local_data = []
            for (inputs, targets) in batch:
                inputs = inputs.cuda()
                targets = targets.cuda()
                inputs = inputs.float().sub_(mean).div_(std)
                if one_hot:
                    targets = expand(num_classes, torch.float, targets)
                if mixup_alpha != 0.0:
                    inputs, targets = mixup(mixup_alpha, inputs, targets)
                parallel_local_data.append([inputs, targets])
            return parallel_local_data

    # TODO: Error: can't pickle torch.memory_format objects by multi-processing iterator
    #collate_fn=partial(fast_collate, memory_format),
    if is_adaptive_training:
        train_loader = jabas.data.AdaptiveDataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            batch_fn=prefetch_func,
            loading_once=False,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = iidp.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            batch_fn=prefetch_func,
            loading_once=True,
            shuffle=(train_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )

    return train_loader


def get_pytorch_val_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    workers=5,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    valdir = os.path.join(data_path, "val")
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    )

    # if torch.distributed.is_initialized():
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    # else:
    #     val_sampler = None

    # collate_fn=partial(fast_collate, memory_format),
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )

    return PrefetchedWrapper(val_loader, 0, num_classes, one_hot), len(val_loader)


class SynteticDataLoader(object):
    def __init__(
        self,
        batch_size,
        num_classes,
        num_channels,
        height,
        width,
        one_hot,
        memory_format=torch.contiguous_format,
    ):
        input_data = (
            torch.empty(batch_size, num_channels, height, width)
            .contiguous(memory_format=memory_format)
            .cuda()
            .normal_(0, 1.0)
        )
        if one_hot:
            input_target = torch.empty(batch_size, num_classes).cuda()
            input_target[:, 0] = 1.0
        else:
            input_target = torch.randint(0, num_classes, (batch_size,))
        input_target = input_target.cuda()

        self.input_data = input_data
        self.input_target = input_target

    def __iter__(self):
        while True:
            yield self.input_data, self.input_target


def get_syntetic_loader(
    data_path,
    batch_size,
    num_classes,
    one_hot,
    start_epoch=0,
    workers=None,
    _worker_init_fn=None,
    memory_format=torch.contiguous_format,
):
    return (
        SynteticDataLoader(
            batch_size, num_classes, 3, 224, 224, one_hot, memory_format=memory_format
        ),
        -1,
    )
