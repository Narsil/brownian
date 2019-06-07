import os
import torch
from torch.utils.data import Dataset, Subset
from collections import OrderedDict
from bisect import bisect_right
import random


class BrownDataset(Dataset):
    def __init__(self, ctx_size):
        self.data = (
            torch.clamp(
                torch.normal(mean=torch.ones(100000), std=100).long(), -500, 500
            )
            + 500
        )
        self.data2 = (
            torch.clamp(torch.normal(mean=torch.ones(100000), std=10).long(), -500, 500)
            + 500
        )
        self.ctx_size = ctx_size

    def __len__(self):
        return self.data.shape[0] - self.ctx_size

    def __getitem__(self, i):
        if random.random() < 0.5:
            return self.data[i : i + self.ctx_size]
        else:
            return self.data2[i : i + self.ctx_size]


def get_sizes(dataset, proportions):
    n = len(dataset)
    total = sum(proportions)
    weights = [p / total for p in proportions]

    sizes = [int(w * n) for w in weights]
    sizes[-1] = n - sum(sizes[:-1])
    return sizes


def random_split(dataset, proportions):
    sizes = get_sizes(dataset, proportions)
    return torch.utils.data.random_split(dataset, sizes)


def split(dataset, proportions):
    sizes = get_sizes(dataset, proportions)
    subsets = []
    offset = 0
    for n in sizes:
        subset = Subset(dataset, range(offset, offset + n))
        subsets.append(subset)
        offset += n
    return subsets
