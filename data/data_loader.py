from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random 


class VizDoomData(Dataset):
    def __init__(self, data,):
        