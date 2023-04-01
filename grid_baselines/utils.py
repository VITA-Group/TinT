
import h5py
import numpy as np
import numpy.linalg as la
import time, os
import pickle
import copy
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from glob import glob


def get_args():
    set_seed(61799)


    args = D()
    args.device                 =       torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.data_dir               =       './preprocessed_data'
    args.city                   =       'BERLIN'
    args.region                 =       'DT'
    args.batch_size             =       16
    args.num_workers            =       8
    args.pin_mem                =       True


    args.berlin_datadir         =       'preprocessed_data/BERLIN/training/*'
    args.use_CVUnfold_or_irregularCell = 0
    args.has_PE                 =       False
    if args.use_CVUnfold_or_irregularCell==1:
        args.N1                 =       8           # for CNN, it is 8
        args.Tire               =       2
        args.N_spatial_kernel   =       (args.Tire+1)*args.Tire/2 * args.N1 + 1

    else:
        args.spatial_kernel_side_length  =  3   # should be odd

    args.model_name             =       'UNet3D'


    args.optfun                 =       [torch.optim.SGD, torch.optim.Adam][0]
    args.lr                     =       0.005
    args.epochs                 =       100


    args.Time_kernel            =       3   # must be odd

    args.dim_feature            =       8

    args.num_res_blocks         =       3
    args.res_block_thickness    =       2

    args.num_layers = args.res_block_thickness * args.num_res_blocks

    args.degrade_to_CNN_kernel  =       False
    args.dim_head               =       2
    args.num_head               =       3
    args.dim_hidden             =       args.num_head * args.dim_head
    args.kernel_neurons         =       [args.dim_hidden, int(args.dim_hidden*1.5), args.dim_hidden]
    print('\n\ntune hyperparams here: if args.kernel_neurons==[args.dim_hidden, args.dim_hidden] (only one layer linear projection), then GCT reduce to CNN. If memory/speed doesnot permit, we can do several layers of CNN, several layers of GCT mixed.\n\n')




    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # 
    #                            data
    # 
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # args.city_2_crop_loc = {'BERLIN#1': ((10,40),(350,390)) }    # ((H_low,H_high),(W_low,W_high))
    # args.city_2_crop_loc = {'BERLIN#1': ((10,20),(350,360)) }    # ((H_low,H_high),(W_low,W_high))
    # args.city_2_crop_loc = {'BERLIN#1': ((10,20),(350,360)) }    # ((H_low,H_high),(W_low,W_high))
    args.city_2_crop_loc = {'BERLIN#1': ((0,128),(0,128)) }    # ((H_low,H_high),(W_low,W_high))
    args.city_2_mask = {}
    for k,v in args.city_2_crop_loc.items():
        crop_shape = [v[0][1]-v[0][0], v[1][1]-v[1][0]]
        args.city_2_mask[k] = np.random.randint(0,2, crop_shape)



    return args
        
def load_1file(fname, args):
    # This function loads the h5 file into npy file to be fed to GCT later.
    # originally loaded shape:  [N,H,W,C], [288=total_time, 495=H, 436=W, 8=C]
    # output shape should be:   [N,H,W,C]
    f = h5py.File(fname, 'r')
    arr = np.array(f['array'])
    
    # please modify here for cropping. -Wenqing
    if 'BERLIN' in fname:
        keyword = 'BERLIN#1'

    if args.city_2_crop_loc[keyword] is not None:
        (Hlo,Hhi), (Wlo,Whi) = args.city_2_crop_loc[keyword]
        croped_arr = arr[:,Hlo:Hhi,Wlo:Whi,:]

        return croped_arr
    else:
        return croped_arr


class SimpleDataloader:
    # This class load data stored in the given folder.
    # return dtype is numpy.ndarray
    # It can manage multi-folders through 'len(folder_list_to_glob)>1'
    class SimpleIterator:
        def __init__(self, args, folder_list_to_glob):
            self.args = args
            self.cnt = 0
            self.maxcnt = 4
            self.folder_list_to_glob = folder_list_to_glob
            self.folder0 = folder_list_to_glob[0]

            self.files0 = glob(self.folder0)

            # self.scales = [2, 60,]

            return


        def __next__(self):
            # ---- customize here ----
            # This function returns (x,y_ground_truth)

            if self.cnt>=self.maxcnt:
                raise StopIteration
            # print('loading data: ', self.cnt)
            file = self.files0[self.cnt]
            arr = load_1file(file, self.args)

            # M = 1
            # arr = arr[::M]

            self.cnt += 1

            # time_len = arr.shape[0]
            time_len = 128
            return arr[:time_len//2, ...], arr[time_len//2:time_len, ...]

    def __init__(self, args, folder_list_to_glob, train_or_test='train'):
        self.args = args
        self.folder_list_to_glob = folder_list_to_glob


    def __iter__(self):
        return self.__class__.SimpleIterator(self.args, self.folder_list_to_glob)




    def __len__(self):
        # ---- customize here ----
        return -1.5

def set_seed(random_seed):
    import torch
    import numpy as np
    import random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


class D:
    def __repr__(self):
        values = []
        def append_values_iterative(obj, prefix=''):
            for att in dir(obj):
                if type(eval(f'obj.{att}')) is D:
                    values.append(prefix + f'{att:16} = \n')
                    append_values_iterative(eval(f'obj.{att}'), prefix='\t')
                elif not att.startswith('__'):
                    v = eval(f'obj.{att}')
                    values.append(prefix + f'{att:16} = {v}\n')
            values.append('\n')
            return
        append_values_iterative(self)
        values = ''.join(values)
        sep = '-'*40 + '\n'
        reprs = sep + values
        return reprs



def viz(net, ttl=''):
    viz_ = []
    flop_est = 0
    for i, (name, p) in enumerate(net.named_parameters()):
        print(f'{name:36}  {list(p.size())}')
        _size = list(p.size())
        viz_.append((name, p.numel(), _size))
        if len(_size)==2: flop_est += _size[0]*_size[1]

    ttl = str(type(net)) if ttl=='' else ttl
    print(f'\nAbove is viz for: {ttl}.\n\tDevice is: {p.device}\n\tN_groups = {len(viz_)}\n\tTotal params = {numParamsOf(net)}\n\tMLP FLOP ~= {flop_est}')
    
    return 


def numParamsOf(net):
    return sum(param.numel() for param in net.parameters())


def getMLP(neurons, activation=nn.GELU, bias=True, dropout=0.1, last_dropout=False, normfun='layernorm'):
    # How to access parameters in module: replace printed < model.0.weight > to < model._modules['0'].weight >
    # neurons: all n+1 dims from input to output
    # len(neurons) = n+1
    # num of params layers = n
    # num of activations = n-1
    if len(neurons) in [0,1]:
        return nn.Identity()
    if len(neurons) == 2:
        return nn.Linear(*neurons)

    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[i+1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[i+1])
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, nn.Dropout(dropout)])
    
    nn_list.extend([nn.Linear(neurons[n-1], neurons[n], bias=bias)])
    if last_dropout:
        nn_list.extend([nn.Dropout(dropout)])
    return nn.Sequential(*nn_list)

def check_nan(x):
    return torch.any(torch.isnan(x))


def check_inf(x):
    return torch.any(torch.isinf(x))
