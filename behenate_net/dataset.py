import os
import random
import numpy as np
import torch
import logging

from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BeHenateDataset(Dataset):
    def __init__(self, data_list, size_sample, trans_list         = None,
                                               normalizes_data    = False,
                                               prints_cache_state = False):
        super().__init__()

        self.data_list          = data_list
        self.size_sample        = size_sample
        self.trans_list         = trans_list
        self.normalizes_data    = normalizes_data
        self.prints_cache_state = prints_cache_state

        self.idx_sample_list    = self.build_dataset()
        self.dataset_cache_dict = {}

        return None


    def __len__(self): return self.size_sample


    def build_dataset(self):
        data_list   = self.data_list
        size_sample = self.size_sample

        size_data_list = len(data_list)
        candidate_list = range(size_data_list)
        idx_sample_list = random.choices(candidate_list, k = size_sample)

        return idx_sample_list


    def get_data(self, idx):
        normalizes_data = self.normalizes_data
        trans_list      = self.trans_list

        idx_sample = self.idx_sample_list[idx]

        img, center, metadata = self.data_list[idx_sample]

        if trans_list is not None:
            for trans in trans_list:
                img, center = trans(img, center)

        if normalizes_data:
            img = (img - img.mean()) / img.std()

        return img, center, metadata


    def __getitem__(self, idx):
        img, center, metadata = self.dataset_cache_dict[idx] \
                                if idx in self.dataset_cache_dict \
                                else self.get_data(idx)

        return img[None,], center, metadata


    def cache_dataset(self, idx_list = []):
        if not len(idx_list): idx_list = range(self.size_sample)
        for idx in idx_list:
            if idx in self.dataset_cache_dict: continue

            if self.prints_cache_state:
                print(f"Cacheing data point {idx}...")

            img, center, metadata = self.get_data(idx)
            self.dataset_cache_dict[idx] = (img, center, metadata)

        return None
