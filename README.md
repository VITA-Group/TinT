# TinT
This repository is the official implementation for the paper "Integrating the Urban Science with Deep Leaning for City-wide Network Congestion Prediction" by Wenqing Zheng, Hao (Frank) Yang, Jiarui Cai, Peihao Wang, Xuan Jiang, Simon Du, Yinhai Wang, and Zhangyang Wang


![](images/FIG1.png "Problem Settings")

![](images/Figure2.png "Proposed Method")



## Dataset Download
We open source our [processed data](https://drive.google.com/drive/folders/1NXnkVIIVHWITdx4EF6y7k2bDimsu8j7t?usp=sharing)


In the required data, two only big size files are < adjmat.pkl > and < speed.h5 >. SD is the smallest dataset.

## Run experiments
Note that both graph and grid modalities share core implementation, with minor differenes in the data processing APIs.

Take the Graph modality for example. After the data is downloaded, cd into the graph folder `cd TinT_graph_modality`. 

Then run `python train_TinT.py --config TinT_configs.conf`.

<!-- 
## (Optional) Data preprocessing

You can also optionally checkout [original data](https://drive.google.com/drive/folders/1u60jmadoMvDe8WZnUItFml6uQuuWK30h?usp=sharing) and process them.

Scripts for preprocessing, if downloading original data:

- we use 28800 (1 record per 5min x 100 days) sample points per city.
- split the dataset to three sub datasets by sensor id, and re-index each city's sensors (start from 0). The purposes of re-indexing are to fit the baseline easier.

```shell
  python prepData_splitCity.py
  # outputs
  # 1. Sensor_lookup_CITY.csv a lookup table for old and new senser id
  # 2. distance_CITY.csv a distance table for sensors in the city, id are in new index
  # 3. CITY.npz data of that city
```
- prepare dataset, train/test/val=0.6/0.2/0.2, each data clip length=12 (1 hour). [loading function](./lib/utils.py#L221)
```
  python preparedata.py --config configurations/prepData_TinT_sd.conf
  python preparedata.py --config configurations/prepData_TinT_sf.conf
  python preparedata.py --config configurations/prepData_TinT_la.conf
``` -->



## How to prepare data for graph modality GCT:

1. download data, as specified in `TinT_configs.conf`

2. manually generate `adj_matrix`  for new dataset: in the main function, modify and save `adj_mx` as `np.save('xxx.npy', adj_mx)`

3. configure `args.dim_hidden_A`: in `TinT.py`, in the function `def tokenizer` print the last line and modify.

## Meanings in the 4 static files in the `.conf`:

`adj_filename`: can be deleted if the `adj_filename` is already there; In order to delete, need to go to main train file and comment `adj_mx, distance_mx = get_adjacency_matrix` and corresponding lines.

`graph_signal_matrix_filename`: will be used in `lib/utils.py` (search for `file_data = np.load(graph_signal_matrix_filename)`)

- in this file, make sure to load 8 entities:
  - train_x; train_target; val_x; val_target; test_x; test_target; mean; std
- how to unify when the output channel is >1: in the main train file, search for `.unsqueeze(-1)` , there are four of them; if label data has dim_channel >1, then you can comment these 4 lines.

`fname_locs`: search for `fname_locs` in `model/gen_locaware_kernel.py`. Make sure it is comparible with `load_locs()`. Desired output is: np.ndarray, shape = [N,2], means node locations.

`adj_matrix`: is an `.npy` file; shape = [N,N]. is manually saved; see above (2.)

 