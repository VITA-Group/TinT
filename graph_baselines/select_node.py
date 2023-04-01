import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

CITY = 'BANGKOK'
num_day = 60

root = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY
data_root = root + '/training/'
static_path  =root + '/' + CITY+'_static.h5'
save_root = root + '/selected_node_process/'
if not os.path.exists(save_root):
    os.mkdir(save_root)

ch_name = ['NE_Volume', 'NE_Average Speed', 'SE_Volume', 'SE_Average Speed', 'SW_Volume', 'SW_Average Speed', 'NW_Volume', 'NW_Average Speed']

valid_node_final = np.asarray(np.load(os.path.join(root, 'selected_node_final.npy')))
print(valid_node_final.shape)

data_list = sorted(os.listdir(data_root))[:num_day]

for file in data_list:
    print(file)
    with h5py.File(os.path.join(data_root, file), 'r') as f:
        data = np.asarray(f['array'])
        ts, h, w, c = data.shape

    for [i, j] in valid_node_final:
        data[:, i, j, :] = -1

    with h5py.File(os.path.join(save_root, file), 'w') as hf:
        hf.create_dataset("array",  data=data)