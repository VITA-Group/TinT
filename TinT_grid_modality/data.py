
import h5py
import numpy as np
import numpy.linalg as la
import math, time, random, os
import pickle
import copy
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import matplotlib.pyplot as plt
from glob import glob

def _convert_dataset(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

        print('Convert h5 files to npy ...')
        file_names = os.listdir(src_dir)
        for filename in tqdm(file_names):
            if not filename.endswith('.h5'):
                continue
            f = h5py.File(os.path.join(src_dir, filename), 'r')
            arr = np.array(f['array'])
            np.save(os.path.join(dst_dir, filename+'.npy'), arr)

class TrafficDatasetBase:

    def __init__(self, root_dir, city, region, split, crop_size=(32, 32), t_len=12, t_stride=1):
        super().__init__()

        self.root_dir = os.path.join(root_dir, city, 'npy')
        _convert_dataset(os.path.join(root_dir, city, 'training'), self.root_dir)

        self.H, self.W = crop_size
        self.t_len, self.t_stride = t_len, t_stride

        if city == 'BERLIN':
            if region == 'DT':
                self.crop_loc = [185, 310]
            elif region == 'Rural':
                self.crop_loc = [119, 68]
        elif city == 'ISTANBUL':
            if region == 'DT':
                self.crop_loc = [126, 273]
            elif region == 'Rural':
                raise ValueError(f'Region {region} is not supported in city {city}')
        elif city == 'BANGKOK':
            if region == 'DT':
                self.crop_loc = [220, 290]
            elif region == 'Rural':
                raise ValueError(f'Region {region} is not supported in city {city}')
        else:
            raise ValueError('Unknown city:', city)

        file_names = os.listdir(self.root_dir)
        file_names = sorted([fn for fn in file_names if fn.endswith('.npy')])

        if split == 'train':
            self.file_names = file_names[:math.floor(len(file_names) * 0.7)]
        elif split == 'valid':
            self.file_names = file_names[math.floor(len(file_names) * 0.7):math.floor(len(file_names) * 0.8)]
        elif split == 'test':
            self.file_names = file_names[math.floor(len(file_names) * 0.8):]


        # we assume each file has the same timestamp length
        arr = np.load(os.path.join(self.root_dir, self.file_names[0]), mmap_mode='r')
        self.times_per_file = arr[::self.t_stride].shape[0] - 2*self.t_len + 1
        arr._mmap.close()

class TrafficDatasetIterator:

    def __init__(self, file_names, shuffle, crop_bbox, t_len, t_stride):
        self.file_names = file_names
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.file_names)

        self.crop_bbox = crop_bbox
        self.t_len, self.t_stride = t_len, t_stride

        self.idxs_in_file = None
        self.arr = None

        self.file_idx = 0
        self.time_idx = 0

    def __next__(self):
        if self.arr is None:
            if self.file_idx >= len(self.file_names):
                raise StopIteration()

            self.arr = np.load(self.file_names[self.file_idx])
            (x, y), (h, w) = self.crop_bbox
            self.arr = self.arr[::self.t_stride, x:x+h, y:y+w].astype(np.float32) / 255.

            self.idxs_in_file = np.arange(self.arr.shape[0] - 2*self.t_len + 1)
            if self.shuffle:
                np.random.shuffle(self.idxs_in_file)
            self.time_idx = 0
            self.file_idx += 1

        i = self.idxs_in_file[self.time_idx]
        x = self.arr[i:i+self.t_len] # [T, H, W, C]
        dec = self.arr[i+self.t_len-1:i+self.t_len*2-1] # [T, H, W, C]
        y = self.arr[i+self.t_len:i+self.t_len*2] # [T, H, W, C]

        x = x.reshape(x.shape[0], -1, x.shape[-1]) # [T, H * W, C]
        dec = dec.reshape(dec.shape[0], -1, dec.shape[-1]) # [T, H * W, C]
        y = y.reshape(y.shape[0], -1, y.shape[-1]) # [T, H * W, C]

        x = x.transpose(1, 0, 2)
        dec = dec.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)

        self.time_idx += 1

        if self.time_idx >= len(self.idxs_in_file):
            self.arr = None
            self.idxs_in_file = None
            self.time_idx = 0

        return x, dec, y

class IteratableTrafficDataset(TrafficDatasetBase, IterableDataset):
    
    def __init__(self, root_dir, city, region, split, shuffle, crop_size=(32, 32), t_len=12, t_stride=1):
        
        TrafficDatasetBase.__init__(self, root_dir, city, region, split, crop_size, t_len, t_stride)
        IterableDataset.__init__(self)

        self.shuffle = shuffle

    def __len__(self):
        return self.times_per_file * len(self.file_names)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self.file_names)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.file_names) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_names))
        file_names = [os.path.join(self.root_dir, f) for f in self.file_names[iter_start:iter_end]]
        return TrafficDatasetIterator(file_names, self.shuffle, (self.crop_loc, (self.H, self.W)),
            self.t_len, self.t_stride)

class IndexableTrafficDataset(TrafficDatasetBase, Dataset):
    
    def __init__(self, root_dir, city, region, split, crop_size=(32, 32), t_len=12, t_stride=1):
        TrafficDatasetBase.__init__(self, root_dir, city, region, split, crop_size, t_len, t_stride)
        Dataset.__init__(self)

    def __len__(self):
        return self.times_per_file * len(self.file_names)

    def __getitem__(self, idx):
        file_idx = idx // self.times_per_file
        idx_in_file = idx % self.times_per_file

        file_path = os.path.join(self.root_dir, self.file_names[file_idx])
        arr = np.load(file_path, mmap_mode='r')

        r0, c0 = self.crop_loc
        r1, c1 = r0 + self.H, c0 + self.W
        cropped_arr = arr[::self.t_stride, r0:r1, c0:c1, :]

        x = np.copy(arr[idx_in_file : idx_in_file+self.t_len])
        dec = self.arr[idx_in_file+self.t_len-1:idx_in_file+self.t_len*2-1] # [T, H, W, C]
        y = np.copy(arr[idx_in_file+self.t_len : idx_in_file+self.t_len*2])

        x = x.reshape(x.shape[0], -1, x.shape[-1]) # [T, H * W, C]
        dec = dec.reshape(dec.shape[0], -1, dec.shape[-1]) # [T, H * W, C]
        y = y.reshape(y.shape[0], -1, y.shape[-1]) # [T, H * W, C]

        x = x.transpose(1, 0, 2)
        dec = dec.transpose(1, 0, 2)
        y = y.transpose(1, 0, 2)

        arr._mmap.close()

        return x, dec, y
