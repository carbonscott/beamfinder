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
                                               prints_cache_state = False,
                                               uses_frac_center   = False,
                                               mpi_comm           = None,):
        super().__init__()

        self.data_list          = data_list
        self.size_sample        = size_sample
        self.trans_list         = trans_list
        self.normalizes_data    = normalizes_data
        self.prints_cache_state = prints_cache_state
        self.uses_frac_center   = uses_frac_center
        self.mpi_comm           = mpi_comm

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

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
        normalizes_data  = self.normalizes_data
        trans_list       = self.trans_list
        uses_frac_center = self.uses_frac_center

        idx_sample = self.idx_sample_list[idx]

        img, center, metadata = self.data_list[idx_sample]

        if trans_list is not None:
            for trans in trans_list:
                img, center = trans(img, center)

        if normalizes_data:
            img = (img - img.mean()) / img.std()

        if uses_frac_center:
            size_y, size_x = img.shape[-2:]
            cy, cx = center
            cy_frac = cy / size_y
            cx_frac = cx / size_x

            center = (cy_frac, cx_frac)

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

            print(f"Cacheing data point {idx}...", flush = True)

            if self.prints_cache_state:
                print(f"Cacheing data point {idx}...")

            img, center, metadata = self.get_data(idx)
            self.dataset_cache_dict[idx] = (img, center, metadata)

        return None


    def mpi_cache_dataset(self, mpi_batch_size = 1):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from .utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        global_idx_list = range(self.size_sample)

        # Divide all indices into batches and go through them...
        batch_idx_list = split_list_into_chunk(global_idx_list, max_num_chunk = mpi_batch_size)
        for batch_seqi, idx_list in enumerate(batch_idx_list):
            # Split the workload...
            idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

            # Process chunk by each worker...
            # No need to sync the dataset_cache_dict across workers
            dataset_cache_dict = {}
            if mpi_rank != 0:
                if mpi_rank < len(idx_list_in_chunk):
                    idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                    dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

                mpi_comm.send(dataset_cache_dict, dest = 0, tag = mpi_data_tag)

            if mpi_rank == 0:
                print(f'[[[ MPI batch {batch_seqi} ]]]', flush = True)

                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)
                self.dataset_cache_dict.update(dataset_cache_dict)

                for i in range(1, mpi_size, 1):
                    dataset_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)
                    self.dataset_cache_dict.update(dataset_cache_dict)

        return None


    def _mpi_cache_data_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        dataset_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in dataset_cache_dict: continue

            print(f"Cacheing data point {idx}...", flush = True)

            img, center, metadata = self.get_data(idx)
            dataset_cache_dict[idx] = (img, center, metadata)

        return dataset_cache_dict
