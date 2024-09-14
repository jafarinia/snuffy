#!/bin/bash

base_path=$(pip show timm | grep Location | cut -d' ' -f2)

sed -i "s/from torch._six import container_abcs/import torch\nTORCH_MAJOR=int(torch.__version__.split('.')[0])\nTORCH_MINOR=int(torch.__version__.split('.')[1])\n\nif TORCH_MAJOR==1 and TORCH_MINOR<8:\n    from torch._six import container_abcs\nelse:\n    import collections.abc as container_abcs/" "$base_path/timm/models/layers/helpers.py"

echo "Modified timm==0.3.2 for compatibility with PyTorch 1.8+: https://github.com/huggingface/pytorch-image-models/issues/420"
