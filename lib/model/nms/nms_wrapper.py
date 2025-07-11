# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.utils.config import cfg
from model.nms.nms_cpu import nms_cpu

# Try to import GPU version, but handle failures gracefully
nms_gpu = None
if torch.cuda.is_available():
    try:
        from model.nms.nms_gpu import nms_gpu
    except ImportError:
        # GPU version not available or incompatible
        nms_gpu = None

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    
    # Use CPU if forced, GPU not available, or GPU import failed
    if force_cpu or nms_gpu is None:
        return nms_cpu(dets, thresh)
    else:
        return nms_gpu(dets, thresh)
