import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append('/data1/sch/EG3D-projector-master/eg3d/projector/PTI')
from configs import global_config, paths_config
from utils.data_utils import make_dataset, make_dataset_label


class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path, c_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')

        if self.source_transform:
            from_im = self.source_transform(from_im)
        
        c = np.load(c_path)
        c = np.reshape(c, (1, 25))
        c = torch.FloatTensor(c).to(global_config.device)

        return fname, from_im, c

class LabelsDataset(Dataset):

    def __init__(self, source_root):
        self.source_paths = sorted(make_dataset_label(source_root))
        # length = len(self.source_paths)
        # i = 1

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, w_path, c_path = self.source_paths[index]
        # w = np.load(from_path)
        w = torch.from_numpy(np.load(w_path)).to(global_config.device)
        c = np.load(c_path)
        c = np.reshape(c, (1, 25))
        c = torch.FloatTensor(c).to(global_config.device)

        return fname, w, c
