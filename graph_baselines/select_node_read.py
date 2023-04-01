import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

CITY = 'BANGKOK'

data_root = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY+ '/training/'
static_path  = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY+ '/' + CITY+'_static.h5'
save_root = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY+ '/selected_node.npy'
save_root_final = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY+ '/selected_node_final.npy'
save_root_img = '/ssd1/peihao/GCT/grid_baselines/preprocessed_data/'+ CITY+ '/selected_node_final.png'
time_thresh = 6
day = 7
volume_thresh = 3

ch_name = ['NE_Volume', 'NE_Average Speed', 'SE_Volume', 'SE_Average Speed', 'SW_Volume', 'SW_Average Speed', 'NW_Volume', 'NW_Average Speed']

data_list = sorted(os.listdir(data_root))
week_data = []

for file in data_list[:day]:
    print(file)
    with h5py.File(os.path.join(data_root, file),'r') as f:
        data = np.asarray(f['array'])
        ts, h, w, c = data.shape
        week_data.append(data)

week_data = np.asarray(week_data)

valid_node = []
for i in range(h):
    for j in range(w):
        slot_data = week_data[:, :, i, j, :].squeeze()
        flatten_slot_data =  slot_data.reshape(-1, slot_data.shape[-1])
        #empty_data = np.asarray([[0]*8]*24)
        for x in range(len(flatten_slot_data)):
            if (flatten_slot_data[x:x + time_thresh]).sum() != 0:
                if np.max(flatten_slot_data[x:x + time_thresh][:,::2]) > volume_thresh:
                   valid_node.append([i,j])
                   break
np.save(save_root, np.asarray(valid_node))
print(np.asarray(valid_node).shape)

valid_node = np.load(save_root)
with h5py.File(static_path, 'r') as f:
    static_data = np.asarray(f['array'])[1:]
valid_node_final = []
for [i,j] in valid_node:
    if static_data[:,i,j].sum() !=0:
        valid_node_final.append([i,j])

np.save(save_root_final, np.asarray(valid_node_final))
print(np.asarray(valid_node_final).shape)
plt.figure()
glid = np.zeros((h,w))
for [i,j] in valid_node:
    glid[i,j] = 0.5
for [i,j] in valid_node_final:
    glid[i,j] += 0.5
plt.imshow(glid)
plt.imsave(save_root_img, glid)
plt.close()