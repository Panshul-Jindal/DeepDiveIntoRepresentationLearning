# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
set -e
cd "$(dirname "$0")"

# Override any of these from the environment, e.g.
#   VOCDIR=/path/to/VOC2007 ./eval_voc_classif.sh
VOCDIR="${VOCDIR:-../data/VOCdevkit/VOC2007}"
MODELROOT="${MODELROOT:-./deepcluster_models}"
MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"
WORKERS="${WORKERS:-8}"
NIT="${NIT:-80000}"

PYTHON="../.venv/bin/python"

# with training the batch norm
# 72.0 mAP
$PYTHON eval_voc_classif.py --vocdir "$VOCDIR" --model "$MODEL" --split trainval \
  --fc6_8 1 --train_batchnorm 1 --workers "$WORKERS" --nit "$NIT"

# without training the batch norm
# 70.4 mAP
$PYTHON eval_voc_classif.py --vocdir "$VOCDIR" --model "$MODEL" --split trainval \
  --fc6_8 1 --train_batchnorm 0 --workers "$WORKERS" --nit "$NIT"
