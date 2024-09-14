# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import timm_modified as timm

assert timm.__version__ == "0.3.2"  # version check
import timm_modified.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

import math
import sys
from typing import Iterable
import util.lr_sched as lr_sched
import ast

import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("WE ONLY WANT GPU")

# ##loading previous data when loading from a saved model part1 (only uncomment this code if needed)
# wandb.init(project="mae_adapter", id='mp4kbcv2')
# run = wandb.Api().run(f"{wandb.run.path}/{wandb.run.id}")
# train_step_loss_logs = [log for log in run.scan_history(keys=['train/step_loss'])]
# valid_step_loss_logs = [log for log in run.scan_history(keys=['valid/step_loss'])]
# train_step_lr_logs = [log for log in run.scan_history(keys=['train/step_lr'])]
# train_step_wd_logs = [log for log in run.scan_history(keys=['train/step_wd'])]
# epoch_train_avg_loss_logs = [log for log in run.scan_history(keys=['train/avg_loss'])]
# epoch_valid_avg_loss_logs = [log for log in run.scan_history(keys=['valid/avg_loss'])]
# step_train_logs = [log for log in run.scan_history(keys=['train/step_train'])]
# step_valid_logs = [log for log in run.scan_history(keys=['valid/step_valid'])]
# wandb.finish()


# steps
step_train = 0
step_valid = 0


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', default=1,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr__min_lr__warmup_epochs', type=list, default=[1e-3, 0., 40], metavar='N',
                        help='for sweep')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./out',
                        help='path where to tensorboard log')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # adapter
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--finetune', default=1, type=int, help='finetuning or from scratch')
    parser.add_argument('--adapter_ffn_scalar', default=0.1, type=float, help='the amount adapter side has effect')
    parser.add_argument('--full_checkpoint', default='mae_pretrain_vit_base_full.pth', type=str, help='full checkpoint')
    parser.add_argument('--wandb_run', help='Name for the wandb run')
    parser.add_argument('--train_linears__linears_from_scratch', type=list, default='[1, 1]', metavar='N',
                        help='for sweep and wether train the head linears of encoder and decoder or not and whether train linears from scratch or not')

    return parser


def main(args):
    global step_train
    global step_valid
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_valid = datasets.ImageFolder(os.path.join(args.data_path, 'validation'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            # ye check konam bebinam in suffle dare ya na
        )
        sampler_valid = torch.utils.data.DistributedSampler(
            dataset_valid, num_replicas=num_tasks, rank=global_rank, shuffle=False
            # ye check konam bebinam in suffle dare ya na
        )
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_valid = %s" % str(sampler_valid))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    if args.model == 'mae_vit_base_patch16':
        adapter_d_model = 768
    if args.model == 'mae_vit_large_patch16':
        adapter_d_model = 1024
    model = models_mae.__dict__[args.model](
        norm_pix_loss=bool(int(args.norm_pix_loss)),
        adapter_ffn_layernorm_option="none",
        adapter_ffn_init_option="lora",
        adapter_ffn_scalar=args.adapter_ffn_scalar,
        adapter_ffn_num=args.ffn_num,
        adapter_d_model=adapter_d_model,
    )

    checkpoint = torch.load(args.full_checkpoint, map_location='cpu')
    checkpoint_model = checkpoint['model']
    if args.finetune == 1:
        state_dict = model.state_dict()
        for k in ['decoder_embed.weight', 'decoder_embed.bias', 'decoder_pred.weight', 'decoder_pred.bias']:
            if k in checkpoint_model and (checkpoint_model[k].shape != state_dict[k].shape or (
                    args.train_linears__linears_from_scratch[1] == 1)):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('msg for initial loading:', msg)
    print(model)

    if args.resume != '':
        start_epoch = torch.load(args.resume, map_location='cpu')['epoch']
        # start_epoch = start_epoch #? or start_epoch = args.start_epoch
        cnt = 0
        # for train
        for idx, log in enumerate(train_step_loss_logs):
            train_step_loss = {
                'train/step_loss': log['train/step_loss'],
            }
            train_step_lr = {
                'train/step_lr': train_step_lr_logs[idx]['train/step_lr'],
            }
            train_step_wd = {
                'train/step_wd': train_step_wd_logs[idx]['train/step_wd']
            }
            train_step = {
                'train/step_train': step_train_logs[idx]['train/step_train'],
            }
            wandb.log({**train_step_loss, **train_step_lr, **train_step_wd, **train_step})
            step_train = step_train_logs[idx]['train/step_train']
            if (idx + 1) % len(data_loader_train) == 0:
                epoch_train_avg_loss = {
                    'train/avg_loss': epoch_train_avg_loss_logs[cnt]['train/avg_loss'],
                    'train/epoch': cnt
                }
                cnt += 1
                wandb.log(epoch_train_avg_loss)
            if (cnt - 1) == (start_epoch):
                step_train += 1
                break
        cnt = 0
        # for valid
        for idx, log in enumerate(valid_step_loss_logs):
            valid_step_loss = {
                'valid/step_loss': log['valid/step_loss'],
            }
            valid_step = {
                'valid/step_valid': step_valid_logs[idx]['valid/step_valid'],
            }
            step_valid = step_valid_logs[idx]['valid/step_valid']
            wandb.log({**valid_step_loss, **valid_step})
            if (idx + 1) % len(data_loader_valid) == 0:
                epoch_valid_avg_loss = {
                    'valid/avg_loss': epoch_valid_avg_loss_logs[cnt]['valid/avg_loss'],
                    'valid/epoch': cnt
                }
                cnt += 1
                wandb.log(epoch_valid_avg_loss)
            if (cnt - 1) == (start_epoch):
                min_val_avg_loss = epoch_valid_avg_loss[
                    'valid/avg_loss']  # this is only true for minimum valid_avg_loss epochs
                min_val_avg_loss_epoch = start_epoch  # this is only true for minimum valid_avg_loss epochs
                step_valid += 1
                break

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
        if (bool(args.train_linears__linears_from_scratch[0])) and (
                name in ['decoder_embed.weight', 'decoder_embed.bias', 'decoder_pred.weight', 'decoder_pred.bias']):
            p.requires_grad = True

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    model.to(device)

    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print('eff_batch_size:', eff_batch_size)

    if args.lr is None:  # only base_lr is specified
        args.lr = float(args.blr__min_lr__warmup_epochs[0]) * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    if args.resume == '':
        min_val_avg_loss = np.inf
        min_val_avg_loss_epoch = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            data_loader_valid, optimizer,
            device, epoch, loss_scaler,
            args=args
        )
        print('min_val_avg_loss, train_stats[1]', min_val_avg_loss, train_stats[1], min_val_avg_loss >= train_stats[1])

        if (args.output_dir and (epoch % args.saveckp_freq == 0 or epoch + 1 == args.epochs)):
            # print('epoch before:', epoch)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )
        if epoch == 0:
            min_val_avg_loss = train_stats[1]
            min_val_avg_loss_epoch = epoch
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )
        if min_val_avg_loss_epoch != None and (min_val_avg_loss >= train_stats[1]):
            os.remove(os.path.join(args.output_dir, f'checkpoint-{min_val_avg_loss_epoch}.pth'))
            min_val_avg_loss = train_stats[1]
            min_val_avg_loss_epoch = epoch
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats[0].items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_valid: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    global step_train
    global step_valid
    model.train(True)
    metric_logger_train = misc.MetricLogger(delimiter="  ")
    metric_logger_train.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    metrics_dict_train = {"step_loss": [], "step_lr": [], "step_wd": []}  # Initialize dictionary
    metrics_dict_valid = {"step_loss": [], "step_lr": [], "step_wd": []}  # Initialize dictionary

    # train
    for data_iter_step, (samples, _) in enumerate(metric_logger_train.log_every(data_loader_train, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger_train.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger_train.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)  ##no use for our setup

        # wandb logging
        metrics_dict_train["step_loss"].append(loss.item())
        metrics_dict_train["step_lr"].append(optimizer.param_groups[0]["lr"])
        metrics_dict_train["step_wd"].append(optimizer.param_groups[1]["weight_decay"])
        step_train_metrics = {
            'train/step_loss': loss.item(),
            'train/step_lr': optimizer.param_groups[0]["lr"],
            'train/step_wd': optimizer.param_groups[1]["weight_decay"],
            'train/step_train': step_train,
        }
        wandb.log(step_train_metrics)
        step_train += 1

    # gather the stats from all processes
    metric_logger_train.synchronize_between_processes()
    print("Train averaged stats:", metric_logger_train)

    # valid
    # we don't do model.train(False) because mae uses bn0
    metric_logger_valid = misc.MetricLogger(delimiter="  ")
    header = 'Valid Epoch: [{}]'.format(epoch)

    with torch.no_grad():
        for data_iter_step, (samples, _) in enumerate(
                metric_logger_valid.log_every(data_loader_valid, print_freq, header)):

            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter

            torch.cuda.synchronize()  # ?

            metric_logger_valid.update(loss=loss_value)

            loss_value_reduce = misc.all_reduce_mean(loss_value)  # no use for our setup

            # wandb logging
            metrics_dict_valid["step_loss"].append(loss.item())
            step_valid_metrics = {
                'valid/step_loss': loss.item(),
                'valid/step_valid': step_valid,
            }
            wandb.log(step_valid_metrics)
            step_valid += 1

    # wandb logging train
    epoch_train_metrics = {
        'train/avg_loss': sum(metrics_dict_train["step_loss"]) / len(metrics_dict_train["step_loss"]),
        'train/epoch': epoch
    }

    # wandb logging valid
    epoch_valid_metrics = {
        'valid/avg_loss': sum(metrics_dict_valid["step_loss"]) / len(metrics_dict_valid["step_loss"]),
        'valid/epoch': epoch
    }
    wandb.log({**epoch_train_metrics, **epoch_valid_metrics})

    # gather the stats from all processes
    metric_logger_valid.synchronize_between_processes()
    print("Valid averaged stats:", metric_logger_valid)

    return {k: meter.global_avg for k, meter in metric_logger_train.meters.items()}, sum(
        metrics_dict_valid["step_loss"]) / len(metrics_dict_valid["step_loss"])


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.blr__min_lr__warmup_epochs = ast.literal_eval(''.join(args.blr__min_lr__warmup_epochs))
    args.train_linears__linears_from_scratch = ast.literal_eval(''.join(args.train_linears__linears_from_scratch))
    if args.wandb_run is None:
        print(f'No wandb name generated by us.')

    wandb.init(
        project=f'mae_adapter',
        config={**vars(args)},
    )

    args.output_dir = os.path.join(args.output_dir, wandb.run.name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
    wandb.finish()
