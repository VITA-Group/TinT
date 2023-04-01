
import matplotlib.pyplot as plt
import torch
import numpy as np


def viz_graph(edge_index, pos, color=None):
    def get_edge_index_tuple(edge_index):
        es = tonp(edge_index).T.tolist()
        return [tuple(x) for x in es]
    import networkx as nx
    pos = tonp(pos)
    G = nx.Graph()
    edge_index_tuples = get_edge_index_tuple(edge_index)
    G.add_edges_from(edge_index_tuples)
    plt.figure(figsize=(8, 8))
    
    markersize = 10
    alpha = 0.1
    if color is None:
        color = [0.2]*len(pos)

    nx.draw_networkx_edges(G, pos, alpha=alpha)

    plt.scatter(pos[:,0], pos[:,1], c=color, s=markersize)

    plt.axis("off")
    # plt.show()
    plt.savefig('nx-graph.pdf')
    return


def tonp(arr):
    if type(arr) is torch.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)


def load_locs(fname):
    # desired output is: np.ndarray, shape = [N,2], means node locations.
    locs_list = load_csv(fname) # [['new_id', 'sensor_id', 'latitude', 'longitude'], ['1', '1114228', '32.54429605', '-117.0321781'], ...]
    locs = []
    for i in range(1, len(locs_list)):
        x = locs_list[i]
        locs.append([float(x[2]), float(x[3])])

    return torch.tensor(locs)


def load_csv(fname):
    x = []
    import csv
    csv_reader = csv.reader(open(fname))
    for line in csv_reader:
        x.append(line)
    return x






fname_locs = '/path/to/your/data/folder/Sensor_location_LA_DT.csv'
adj_matrix = '/path/to/your/data/folder/Adj_matrix_LADT.npy'



adj = np.load(adj_matrix)
locs = load_locs(fname_locs)


edge_index = tonp(torch.tensor(adj).to_sparse().indices())

viz_graph(edge_index, locs)
import time
lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
plt.savefig(f'visualize-jan16----{lt2}.pdf')





