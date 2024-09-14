# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Everything seems fine except resuming being a bit non-deterministic which is the same as what official dino does

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer_with_adapter as vits
from vision_transformer_with_adapter import DINOHead
import ast
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise Exception("WE ONLY WANT GPU")

# ## Loading previous data when loading from a saved model part1 (Uncomment this only if you need to)
# wandb.init(project="dino_adapter", id='0wykwarv')
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

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp__warmup_teacher_temp_epochs', default=[0.04, 0], type=list,
                        help="""for sweep""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay__weight_decay_end', type=list, default=[0.04, 0.4], help="""for sweep""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr__warmup_epochs__minlr", default=[0.0005, 10, 1e-6], type=list, help="""for sweep""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path_train', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--data_path_valid', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--finetune', default=1, type=int, help='finetuning or from scratch')
    parser.add_argument('--adapter_ffn_scalar', default=0.1, type=float, help='the amount adapter side has effect')
    parser.add_argument('--full_checkpoint', default='dino_vitbase8_pretrain_full_checkpoint.pth', type=str,
                        help='full checkpoint')
    parser.add_argument('--wandb_run', help='Name for the wandb run')
    parser.add_argument('--resume', default=0, type=int, help='if its 1 then we resume from the last model saved')
    parser.add_argument('--random_head', default=0, type=int,
                        help='if its 1 then train head from scratch, if its 0 then we train head from checkpoint')

    return parser


def train_dino(args):
    global step_train
    global step_valid
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset_train = datasets.ImageFolder(args.data_path_train, transform=transform)
    dataset_valid = datasets.ImageFolder(args.data_path_valid, transform=transform)
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    sampler_valid = torch.utils.data.DistributedSampler(dataset_valid, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_valid,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: there are {len(dataset_train)} images.")
    print(f"Data loaded: there are {len(dataset_valid)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        if args.arch == 'vit_small':
            adapter_d_model = 384
        elif args.arch == 'vit_base':
            adapter_d_model = 768
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            adapter_ffn_layernorm_option="none",
            adapter_ffn_init_option="lora",
            adapter_ffn_scalar=args.adapter_ffn_scalar,
            adapter_ffn_num=args.ffn_num,
            adapter_d_model=adapter_d_model,
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            adapter_ffn_layernorm_option="none",
            adapter_ffn_init_option="lora",
            adapter_ffn_scalar=args.adapter_ffn_scalar,
            adapter_ffn_num=args.ffn_num,
            adapter_d_model=adapter_d_model,
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    if args.finetune == 1:
        full_checkpoint = torch.load(args.full_checkpoint, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.full_checkpoint)
        full_checkpoint_student = full_checkpoint['student']
        full_checkpoint_teacher = full_checkpoint['teacher']
        state_dict_student = student.state_dict()
        state_dict_teacher = teacher.state_dict()
        # student
        for k in ['module.head.mlp.0.weight', 'module.head.mlp.0.bias',
                  'module.head.mlp.2.weight', 'module.head.mlp.2.bias',
                  'module.head.mlp.4.weight', 'module.head.mlp.4.bias',
                  'module.head.last_layer.weight', 'module.head.last_layer.weight_g',
                  'module.head.last_layer.weight_v']:
            k_standard_form = k.split('.', maxsplit=1)[1]
            if args.arch == 'vit_base' and k == 'module.head.last_layer.weight':
                k_standard_form += '_v'
            # print('k_standard_form:', k_standard_form)
            if k in full_checkpoint_student and (full_checkpoint_student[k].shape != state_dict_student[
                k_standard_form].shape or args.random_head == 1):
                print(f"Removing key {k} from pretrained student checkpoint")
                del full_checkpoint_student[k]
        # teacher
        for k in ['head.mlp.0.weight', 'head.mlp.0.bias',
                  'head.mlp.2.weight', 'head.mlp.2.bias',
                  'head.mlp.4.weight', 'head.mlp.4.bias',
                  'head.last_layer.weight', 'head.last_layer.weight_g',
                  'head.last_layer.weight_v']:
            k_standard_form = k
            if args.arch == 'vit_base' and k == 'head.last_layer.weight':
                k_standard_form += '_v'
            # print('k, k_standard_form:', k, k_standard_form)
            if k in full_checkpoint_teacher and (full_checkpoint_teacher[k].shape != state_dict_teacher[
                k_standard_form].shape or args.random_head == 1):
                print(f"Removing key {k} from pretrained teacher checkpoint")
                del full_checkpoint_teacher[k]

        # adjusting weights
        adjusted_student_dict = {}
        for k, v in full_checkpoint_student.items():
            # Adjusting the key by removing 'module.' prefix
            new_key = k[7:] if k.startswith('module.') else k

            # Specific mapping for module.head.last_layer.weight
            if new_key == 'head.last_layer.weight':
                new_key = 'head.last_layer.weight_v'

            adjusted_student_dict[new_key] = v
        # Make sure head.last_layer.weight_g remains unchanged
        if args.arch == 'vit_base':
            adjusted_student_dict['head.last_layer.weight_g'] = student.state_dict()['head.last_layer.weight_g']
        msg_student = student.load_state_dict(adjusted_student_dict, strict=False)

        # adjusting weights
        adjusted_teacher_dict = {}
        for k, v in full_checkpoint_teacher.items():
            # Adjusting the key by removing 'module.' prefix
            new_key = k

            # Specific mapping for module.head.last_layer.weight
            if new_key == 'head.last_layer.weight':
                new_key = 'head.last_layer.weight_v'

            adjusted_teacher_dict[new_key] = v
        # Make sure head.last_layer.weight_g remains unchanged
        if args.arch == 'vit_base':
            adjusted_teacher_dict['head.last_layer.weight_g'] = teacher.state_dict()['head.last_layer.weight_g']
        teacher.load_state_dict(adjusted_teacher_dict, strict=False)

        # freeze all but the head
        for name, p in student.named_parameters():
            if name in msg_student.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for _, p in student.head.named_parameters():  # ?
            p.requires_grad = True

        n_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        float(args.teacher_temp__warmup_teacher_temp_epochs[0]),
        int(args.teacher_temp__warmup_teacher_temp_epochs[1]),
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        float(args.lr__warmup_epochs__minlr[0]) * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        # linear scaling rule
        float(args.lr__warmup_epochs__minlr[2]),
        args.epochs, len(data_loader_train),
        warmup_epochs=int(args.lr__warmup_epochs__minlr[1]),
    )
    wd_schedule = utils.cosine_scheduler(
        float(args.weight_decay__weight_decay_end[0]),
        float(args.weight_decay__weight_decay_end[1]),
        args.epochs, len(data_loader_train),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader_train))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    start_epoch = to_restore["epoch"]
    if args.resume == 1:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            # scheduler=scheduler
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
        start_epoch = to_restore["epoch"]
        # loading previous data for wandb
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
            if cnt == (start_epoch):
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
            if cnt == (start_epoch):
                step_valid += 1
                break

    start_time = time.time()
    print("Starting DINO training !")

    print('start_epoch:', start_epoch)
    for epoch in range(start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                      data_loader_train, data_loader_valid, optimizer, lr_schedule, wd_schedule,
                                      momentum_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                    data_loader_train, data_loader_valid, optimizer, lr_schedule, wd_schedule,
                    momentum_schedule, epoch, fp16_scaler, args):
    global step_train
    global step_valid
    metric_logger_train = utils.MetricLogger(delimiter="  ")
    metric_logger_valid = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    metrics_dict_train = {"step_loss": [], "step_lr": [], "step_wd": []}  # Initialize dictionary
    metrics_dict_valid = {"step_loss": [], "step_lr": [], "step_wd": []}  # Initialize dictionary

    # train
    student.train()
    teacher.train()
    for it, (images, _) in enumerate(metric_logger_train.log_every(data_loader_train, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader_train) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger_train.update(loss=loss.item())
        metric_logger_train.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger_train.update(wd=optimizer.param_groups[0]["weight_decay"])

        # wandb logging
        metrics_dict_train["step_loss"].append(loss.item())
        metrics_dict_train["step_lr"].append(optimizer.param_groups[0]["lr"])
        metrics_dict_train["step_wd"].append(optimizer.param_groups[0]["weight_decay"])
        step_train_metrics = {
            'train/step_loss': loss.item(),
            'train/step_lr': optimizer.param_groups[0]["lr"],
            'train/step_wd': optimizer.param_groups[0]["weight_decay"],
            'train/step_train': step_train,
        }
        wandb.log(step_train_metrics)
        step_train += 1

    # valid
    student.eval()
    teacher.eval()
    with torch.no_grad():
        for it, (images, _) in enumerate(metric_logger_valid.log_every(data_loader_valid, 10, header)):
            # move images to gpu
            images = [im.cuda(non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping validation".format(loss.item()), force=True)
                sys.exit(1)

            # logging
            torch.cuda.synchronize()
            metric_logger_valid.update(loss=loss.item())

            # wandb logging
            metrics_dict_valid["step_loss"].append(loss.item())
            step_valid_metrics = {
                'valid/step_loss': loss.item(),
                'valid/step_valid': step_valid
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
    metric_logger_train.synchronize_between_processes()
    metric_logger_valid.synchronize_between_processes()
    print("Averaged stats:", metric_logger_train)
    print("Averaged stats:", metric_logger_valid)
    # return {k: meter.global_avg for k, meter in metric_logger_valid.meters.items()}
    return {k: meter.global_avg for k, meter in metric_logger_train.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.teacher_temp__warmup_teacher_temp_epochs = ast.literal_eval(
        ''.join(args.teacher_temp__warmup_teacher_temp_epochs))
    args.weight_decay__weight_decay_end = ast.literal_eval(''.join(args.weight_decay__weight_decay_end))
    args.lr__warmup_epochs__minlr = ast.literal_eval(''.join(args.lr__warmup_epochs__minlr))

    if args.wandb_run is None:
        print(f'No wandb name generated by us.')

    wandb.init(
        project=f'dino_adapter',
        config={**vars(args)},
        # mode='offline'
    )
    print(f'*** Run Config *** ')
    utils.pretty_print({**vars(args)})

    args.output_dir = os.path.join(args.output_dir, wandb.run.name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

    wandb.finish()
