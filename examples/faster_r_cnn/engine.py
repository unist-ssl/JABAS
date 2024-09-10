import math
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
from utils import AverageMeter, ProgressMeter


def train_one_epoch(train_loader, trainer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.5f')
    comp_time = AverageMeter('Comp', ':6.5f')
    update_time = AverageMeter('Update', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')
    if args.is_adaptive_training:
        num_batches = len(train_loader.dataset)
    else:
        if args.num_minibatches is not None:
            num_batches = args.num_minibatches
        else:
            num_batches = len(train_loader)
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, comp_time, update_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    trainer.set_model_train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            trainer.main_optimizer, warmup_iters, warmup_factor)

    end = time.time()
    warmup_step = 10 * (trainer.accum_step+1)
    for i, data in enumerate(train_loader):
        if args.num_minibatches is not None and trainer.sync_step > args.num_minibatches:
            break
        is_record_step = ((i+1) % (trainer.accum_step+1) == 0)
        # measure data loading time
        if i >= warmup_step and is_record_step:
            data_time.update(time.time() - end)
            start = time.time()

        trainer.compute(data)
        if i >= warmup_step and is_record_step:
            comp_time.update(time.time() - start)

        # Record loss
        loss = trainer.losses[0].detach()
        losses.update(loss, trainer.local_batch_size)
        # As it has overhead, we check only if convergence procedure
        if not math.isfinite(loss) and args.num_minibatches is None:
            print(f"Loss is {loss}, stopping training")
            exit(1)

        if i >= warmup_step and is_record_step:
            start = time.time()

        # Update parameters
        is_sync_step = trainer.step()

        if epoch == 0 and is_sync_step:
            lr_scheduler.step()

        if i >= warmup_step and is_record_step:
            update_time.update(time.time() - start)

        # measure elapsed time
        if i >= warmup_step and is_record_step:
            batch_time.update(time.time() - end)
        if is_record_step:
            end = time.time()

        if args.is_adaptive_training:
            if is_record_step and (trainer.sync_step % args.print_freq == 0) and \
                    torch.distributed.get_rank() == 0:
                progress.display(train_loader.data_index)
        else:
            if is_record_step and ((train_loader.step_index+1) % args.print_freq == 0):
                # As step starts from 0, printing step+1 is right
                progress.display(train_loader.step_index+1)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or \
            isinstance(model, torch.nn.parallel.IndependentIdenticalDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[INFO] Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
