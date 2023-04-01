import numpy as np
import torch
import math






def build_non_iso_adjs(args_out):
    # returns: a list of sparse tensors

    city = args_out['Data']['adj_filename'][-6:-4] # 'SD'
    device = torch.device(f'cuda:{int(args_out["Training"]["cudaID"])}')
    import ml_collections
    _args = ml_collections.ConfigDict(args_out)

    # ---- load adjs ----
    adjNN = np.load(args_out['Data']['adj_matrix'])
    adjNN = torch.tensor(adjNN).to_sparse().indices()
    _args.edge_index = adjNN
    _args.fname_locs = args_out['Data']['fname_locs']



    _args.Tmax, _args.num_tires_space, _args.num_tires_time = 12, 1, 1
    _args.num_relations = 4

    edge_list = load_multi_kernel_with_csv(_args)
    adjs = []
    for edge_index in edge_list:
        adj_NN = edge_index_to_dense_numpy(edge_index, _args.N_nodes*_args.Tmax )
        adj_NN = torch.tensor(scaled_Laplacian(adj_NN)).float().to_sparse().to(device)
        adjs.append(adj_NN)

    return adjs


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W
    from scipy.sparse.linalg import eigs
    lambda_max = eigs(L, k=1, which='LR')[0].real
    if lambda_max==0: lambda_max=1

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def edge_index_to_dense_numpy(edge_index, num_nodes=None, edge_weight=None):
    # output is N*N numpy array
    if num_nodes is None:
        num_nodes = int(edge_index.max()+1)
    if edge_weight is None:
        v = [1] * edge_index.shape[1]
    adj = torch.sparse_coo_tensor(edge_index, v, size=(num_nodes, num_nodes), device=edge_index.device).to_dense().float().numpy()
    return adj


def load_multi_kernel_with_csv(args):

    locs = load_locs(args.fname_locs)
    args.N_nodes = len(locs)
    
    adj_list = assign_multi_kernel(locs, **args.to_dict())

    # viz_graph(clean_up(adj_list[4], len(locs)), locs)

    return adj_list

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

def clean_up(edge_index, N_total):
    m = (edge_index[0] >= 0) & (edge_index[0] < N_total)
    m &= ((edge_index[1] >= 0) & (edge_index[1] < N_total))
    return edge_index[:, m]

def assign_multi_kernel(locs, edge_index, Tmax, num_relations, num_tires_space, num_tires_time, **w):
    # This function finds and assigns the multi-relational edges, based on node locations.
    # output:
        # edges[num_tires_time]: self-loop
        # edges[num_tires_time + k * (2*num_tires_time+1) ]: current time stamp
    # if num_relations=4, num_tires_time=1, then output len=15, sequence is:
        # [0,1,2]:  loc=center, t=-1,0,1
        # [3,4,5]:  tire=1, relation=1, t=-1,0,1
        # [6,7,8]:  tire=1, relation=2, t=-1,0,1
        # [9,10,11]:   ...
        # [12,13,14]:  ...


    # When num_relations = 1, number of spatial relationships = 1, and reduce to GCN. num_relations must > 0.
    # num_tires_space means counting from

    # num_tires_time means count of temporal kernel, and do not include the center, i.e., if set to 0, it means not aggregating from past/future time slices; if set to 1, it means aggregates from 1 step past and future, len(temporal_kernel) = 3.
    # the total number of relationships should be: 
        # (2*num_tires_time + 1) * (1 + num_tires_space*num_relations)

    # key variables:
        # N: number of spatial nodes
        # N_ST: number of space-time nodes, N_ST = N*Tmax
        # locs: node locations, np.ndarray, shape = [N,2]
        # adj: adj matrix, np.ndarray, shape = [N,N]
        # Tmax: max temporal input length, int
        # num_relations: number of relationship to split spatially; int
        # num_tires_space: number of spatial tires (max hops), int
        # num_tires_time: number of temporal tires, int
        # adj_list: a list of edge_index. The order should be: historical time kernel first, and inner tire spatial kernel first.
        # edge_index: torch.tensor, shape = [2, num_edges], edge_index.max()+1 = N_ST

    N = locs.shape[0]
    N_total = N * Tmax
    src = torch.index_select(locs, 0, edge_index[0]) # [E, 2]
    dst = torch.index_select(locs, 0, edge_index[1]) # [E, 2]
    dx, dy = (dst - src).permute([1, 0]) # [E, 2]
    rad = torch.atan2(dy, dx) + math.pi # range: [0, 2*pi]

    edges = []

    ones = torch.ones(edge_index.shape[-1], dtype=torch.float, device=edge_index.device)
    adj = torch.sparse_coo_tensor(edge_index, ones, (N, N))

    # historical time kernel
    for s in range(-num_tires_time, num_tires_time+1):
        shift_central = torch.stack([torch.arange(N_total), torch.arange(N_total) + s * N], 0)
        edges.append(clean_up(shift_central, N_total))

    for k in range(num_tires_space):

        # inner tire spatial kernel 
        for i in range(num_relations):

            # compute for one time slice
            lower = i * 2 * math.pi / num_relations
            upper = (i+1) * 2 * math.pi / num_relations
            isel = (rad > lower) & (rad < upper) # [E,]
            tire = edge_index[:, isel]

            # repeat along time axis
            edge_tires = []
            for t in range(2*Tmax+1):
                tire_new = torch.stack([tire[0] + N * t, tire[1] + N * t], 0)
                edge_tires.append(tire_new)
            edge_tires = torch.cat(edge_tires, dim=1) # [2, N * (Tmax * 2 + 1)]


            # shift along time axis
            for s in range(-num_tires_time, num_tires_time+1):
                shift_neighor = torch.stack([edge_tires[0], edge_tires[1] + N * s], 0) # [2, N * (Tmax * 2 + 1)]
                edges.append(clean_up(shift_neighor, N_total))

        adj = torch.sparse.mm(adj, adj)
        edge_index = adj.coalesce().indices()

    return edges


class D: pass



def debug():
    args = D()

    args.Tmax = 12
    args.num_relations = 3
    args.num_tires_space = 1
    args.num_tires_time = 1
    args.filename = 'Sensor_SD.csv'

    adj = np.load('Adj_matrix_SD_nov21.npy')
    args.edge_index = torch.tensor(adj).to_sparse().indices()
    
    adj_list = load_multi_kernel_with_csv(args)

    # locs = torch.tensor([[0, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=torch.float32)
    # edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long)
    # edges = assign_multi_kernel(locs, edge_index, args.Tmax, args.num_relations, args.num_tires_space, args.num_tires_time)

    return



if __name__ == '__main__':
    debug()


