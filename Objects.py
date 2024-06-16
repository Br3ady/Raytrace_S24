import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random


def sphere(r,resolution):
    phi = np.linespace(0,np.pi,)
    