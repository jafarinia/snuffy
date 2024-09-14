# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < int(args.blr__min_lr__warmup_epochs[2]):
        lr = args.lr * epoch / int(args.blr__min_lr__warmup_epochs[2])
    else:
        lr = float(args.blr__min_lr__warmup_epochs[1]) + (args.lr - float(args.blr__min_lr__warmup_epochs[1])) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - int(args.blr__min_lr__warmup_epochs[2])) / (
                         args.epochs - int(args.blr__min_lr__warmup_epochs[2]))))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
