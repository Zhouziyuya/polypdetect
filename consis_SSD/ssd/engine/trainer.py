import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist
import random
import numpy as np
import cv2
import sys
from PIL import Image

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger
from ssd.consis_loss import ConsistencyLossLR, ConsistencyLossUD, ConsistencyLossLRUD


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(cfg, model, teacher,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()
    consistency_loss_lr = ConsistencyLossLR(cfg.MODEL.NEG_POS_RATIO)
    consistency_loss_ud = ConsistencyLossUD(cfg.MODEL.NEG_POS_RATIO)
    consistency_loss_lrud = ConsistencyLossLRUD(cfg.MODEL.NEG_POS_RATIO)

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    

    momentum_schedule = cosine_scheduler(0.99, 0.9999, cfg.SOLVER.MAX_ITER, 5000) # warmup_epochs = 5

    for iteration, (images, img_fliplr, img_flipud, img_fliplrud, targets, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # img = images[0].permute(1,2,0).numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('figures/img.png', img)

        # img_fliplr = img_fliplr[0].permute(1,2,0).numpy()
        # img_fliplr = cv2.cvtColor(img_fliplr, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('figures/img_fliplr.png', img_fliplr)

        # img_flipud = img_flipud[0].permute(1,2,0).numpy()
        # img_flipud = cv2.cvtColor(img_flipud, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('figures/img_flipud.png', img_flipud)

        # img_fliplrud = img_fliplrud[0].permute(1,2,0).numpy()
        # img_fliplrud = cv2.cvtColor(img_fliplrud, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('figures/img_fliplrud.png', img_fliplrud)
        # sys.exit(1)

        images = images.to(device)
        img_fliplr = img_fliplr.to(device)
        img_flipud = img_flipud.to(device)
        img_fliplrud = img_fliplrud.to(device)
        targets = targets.to(device)

        detections, loss_dict = model(images, targets=targets)
        detections_t, _ = teacher(images, targets=targets)

        if args.consis_mode == 'lr':
            detections_flip_t, _ = teacher(img_fliplr, targets=targets)
            loss_consis1 = consistency_loss_lr(detections, detections_flip_t)

            detections_flip_s, _ = model(img_fliplr, targets=targets)
            loss_consis2 = consistency_loss_lr(detections_t, detections_flip_s)
        elif args.consis_mode == 'ud':
            detections_t, _ = teacher(img_flipud)
        elif args.consis_mode == 'lrud':
            

            r = random.random()
            if r < 0.25:
                detections_flip_t, _ = teacher(img_fliplr, targets=targets)
                loss_consis1 = consistency_loss_lr(detections, detections_flip_t)

                detections_flip_s, _ = model(img_fliplr, targets=targets)
                loss_consis2 = consistency_loss_lr(detections_t, detections_flip_s)
            elif r>=0.25 and r<0.5:
                detections_flip_t, _ = teacher(img_flipud, targets=targets)
                loss_consis1 = consistency_loss_ud(detections, detections_flip_t)

                detections_flip_s, _ = model(img_flipud, targets=targets)
                loss_consis2 = consistency_loss_ud(detections_t, detections_flip_s)
            elif r>=0.5 and r<0.75:
                detections_flip_t, _ = teacher(img_fliplrud, targets=targets)
                loss_consis1 = consistency_loss_lrud(detections, detections_flip_t)

                detections_flip_s, _ = model(img_fliplrud, targets=targets)
                loss_consis2 = consistency_loss_lrud(detections_t, detections_flip_s)
            else:
                loss_consis1 = 0
                loss_consis2 = 0
            # loss_consis_items = [x+y for x,y in zip(loss_consis_items1, loss_consis_items2)]


        loss_consis = loss_consis1+loss_consis2
        loss_sup = sum(loss for loss in loss_dict.values())
        loss = loss_sup+loss_consis



        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[iteration]  # momentum parameter
            for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if device != "cpu":
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.5f}",
                        '{meters}',
                        'consis loss: {loss_consis:.3f}',
                        "eta: {eta}",
                        'mem: {mem}M',
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        loss_consis=loss_consis,
                        eta=eta_string,
                        mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                    )
                )
            else:
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.5f}",
                        '{meters}',
                        'consis loss: {loss_consis:.3f}',
                        "eta: {eta}",
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        loss_consis=loss_consis,
                        eta=eta_string,
                    )
                )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/consis_loss', loss_consis, global_step=global_step)
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model


def cosine_scheduler(base_value, final_value, total_iter, warmup_iters=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    # warmup_iters = warmup_epochs * niter_per_ep
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(total_iter - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    # assert len(schedule) == epochs * niter_per_ep
    return schedule
