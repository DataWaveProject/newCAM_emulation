import netCDF4 as nc
import numpy as np
import scipy.stats as st
import xarray as xr

import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Required for feeding the data iinto NN.
class myDataset(Dataset):
    def __init__(self, X, Y):
        """
        Parameters:
            X (tensor): Input data.
            Y (tensor): Output data.
        """
        self.features = torch.tensor(X, dtype=torch.float64)
        self.labels = torch.tensor(Y, dtype=torch.float64)

    def __len__(self):
        """Function that is called when you call len(dataloader)"""
        return len(self.features.T)

    def __getitem__(self, idx):
        """Function that is called when you call dataloader"""
        feature = self.features[:, idx]
        label = self.labels[:, idx]

        return feature, label


# The NN model.
class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class FullyConnected(nn.Module):
    def __init__(self, ilev, mean, std):
        super(FullyConnected, self).__init__()
        self.normalization = NormalizationLayer(mean, std)
        self.ilev = ilev

        layers = []
        layers.append(nn.Linear(8 * ilev + 4, 500))
        layers.append(nn.SiLU())
        
        num_layers = 10  # Example: Change this to the desired number of hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(500, 500))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(500, 2 * ilev))
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalization(x)
        return self.linear_stack(x)
