from __future__ import print_function
import torch
import torch.nn.functional as F

import numpy as np

from util.PointNetpp_utils import SetAbstraction

class PointNetpp(nn.Module):
    def __init__(self, num_class: int = 10, normal_channel = True): 
        super(PointNetpp, self).__init__()
        inchannel = 6 if normal_channel else 3
