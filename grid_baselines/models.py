from functools import partial

import torch
import torch.nn as nn
from unet3d import UNet2D, UNet3D
from vit import VisionTransformer, SwinTransformer, VideoTransformer

class TrafficUNet2D(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args

        self.norm_in = nn.BatchNorm2d(args.dim_feature)
        self.unet = UNet(args.dim_feature, args.dim_feature)

        return

    def forward(self, x):

        B, N, H, W, C = x.shape

        x = x.reshape(-1, H, W, C) # [B,N,H,W,C] -> [B*N,C,H,W]

        x = x.permute(0,3,1,2)   # [N,H,W,C] -> [N,C,H,W]
        x = self.norm_in(x)
        
        x = self.unet(x)

        x = x.permute(0,2,3,1)   # [N,C,H,W] -> [N,H,W,C]

        x = x.reshape(B, N, H, W, C) # [B,N,H,W,C] -> [B*N,C,H,W]

        return x

class TrafficUNet2D_LSTM(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.unet = UNet2D(args.dim_feature, args.dim_hidden, f_map=args.dim_hidden, is_segmentation=False)
        self.lstm = nn.LSTM(args.dim_hidden, args.dim_hidden, batch_first=True)
        self.lin_out = nn.Linear(args.dim_hidden, args.dim_feature)

    def forward(self, x):

        x = x.permute(0,4,1,2,3)   # [N,T,H,W,C] -> [N,C,T,H,W]
        x = self.unet(x)
        x = x.permute(0,3,4,2,1)   # [N,C,T,H,W] -> [N,H,W,T,C]

        N, H, W, T, C = x.shape
        x = x.reshape(-1, T, C)
        x, _ = self.lstm(x)
        x = x.reshape(N, H, W, T, C)
        x = x.permute(0,3,1,2,4)   # [N,H,W,T,C] -> [N,T,H,W,C]
        x = self.lin_out(x)

        return x

class TrafficUNet3D(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.unet = UNet3D(args.dim_feature, args.dim_feature, f_map=args.dim_hidden, is_segmentation=False)


    def forward(self, x):

        x = x.permute(0,4,1,2,3)   # [N,T,H,W,C] -> [N,C,T,H,W]
        x = self.unet(x)
        x = x.permute(0,2,3,4,1)   # [N,C,T,H,W] -> [N,T,H,W,C]

        return x

class TrafficViT(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.patch_size = args.patch_size
        self.out_dim = args.dim_feature

        self.vit = VisionTransformer(img_size=(args.crop_size*args.pred_len, args.crop_size), patch_size=args.patch_size, pool='none',
            in_chans=args.dim_feature, num_classes=args.patch_size * args.patch_size * args.dim_feature, embed_dim=args.dim_hidden,
            depth=args.num_layers, num_heads=args.num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def forward(self, x):

        B, T, H, W, Cin = x.shape
        ph = pw = self.patch_size
        h, w = H // ph, W // pw
        Cout = self.out_dim

        x = x.reshape(B, T*H, W, Cin) # [N,T,H,W,C] -> [N,TxH,W,C]
        x = x.permute(0,3,1,2)     # [N,TxH,W,C] -> [N,C,TxH,W]
        x = self.vit(x)            # [N,TxHxW,Cxhxw]
        x = x.reshape(B, T, h, w, ph, pw, Cout) # [N,TxHxW,Cxhxw] -> [N,T,H,W,h,w,C]
        x = x.permute(0,1,2,4,3,5,6) # [N,T,H,W,h,w,C] -> [N,T,H,h,W,w,C]
        x = x.reshape(B, T, h*ph, w*pw, Cout)

        return x

class TrafficSwin(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.patch_size = args.patch_size
        self.out_dim = args.dim_feature
        self.vit = SwinTransformer(img_size=(args.crop_size*args.pred_len, args.crop_size), patch_size=args.patch_size, pool='none',
            in_chans=args.dim_feature, num_classes=args.patch_size * args.patch_size * args.dim_feature, embed_dim=args.dim_hidden,
            depths=[args.num_layers], num_heads=[args.num_heads], window_size=args.window_size, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def forward(self, x):

        B, T, H, W, Cin = x.shape
        ph = pw = self.patch_size
        h, w = H // ph, W // pw
        Cout = self.out_dim

        x = x.reshape(B, T*H, W, Cin) # [N,T,H,W,C] -> [N,TxH,W,C]
        x = x.permute(0,3,1,2)     # [N,TxH,W,C] -> [N,C,TxH,W]
        x = self.vit(x)            # [N,TxHxW,Cxhxw]
        x = x.reshape(B, T, h, w, ph, pw, Cout) # [N,TxHxW,Cxhxw] -> [N,T,H,W,h,w,C]
        x = x.permute(0,1,2,4,3,5,6) # [N,T,H,W,h,w,C] -> [N,T,H,h,W,w,C]
        x = x.reshape(B, T, h*ph, w*pw, Cout)

        return x

class TrafficViViT(nn.Module):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.out_dim = args.dim_feature
        self.vit = VideoTransformer(args.crop_size, args.patch_size, args.crop_size * args.crop_size * args.dim_feature, args.pred_len,
             pool='none', in_channels=args.dim_feature, dim=args.dim_hidden, depth=args.num_layers, heads=args.num_heads)

    def forward(self, x):

        B, T, H, W, _ = x.shape
        C = self.out_dim

        x = x.permute(0,1,4,2,3)   # [N,T,H,W,C] -> [N,T,C,H,W]
        x = self.vit(x)            # [N,T,CxHxW]
        x = x.reshape(B, T, C, H, W)
        x = x.permute(0,1,3,4,2)   # [N,T,C,H,W] -> [N,T,H,W,C]

        return x

class GCT(nn.Module):
    def __init__(self, args):


        super().__init__()
        self.A = nn.Linear(3,3)

        self.num_layers = args.num_layers
        self.has_PE = args.has_PE

        self.args = args
        self.in_proj = nn.Linear(args.dim_feature, args.dim_hidden)
        self.out_proj = nn.Linear(args.dim_hidden, args.dim_feature)

        self.norm_in = nn.BatchNorm2d(args.dim_feature)
        self.norm_layers = nn.ModuleList([nn.BatchNorm2d(args.dim_hidden) for l in range(self.num_layers)])


        _GCTLayer = GCTLayer if args.use_CVUnfold_or_irregularCell==0 else GCTLayer_IrregularCell


        self.layers = nn.ModuleList([_GCTLayer(args) for l in range(self.num_layers)])

        return

    def update_cell_neighbor(self, city_keyword):
        mask = self.args.city_2_mask[city_keyword]
        for layer in self.layers:
            layer.update_cell_neighbor(mask)
        return


    def forward(self, x):
        # input shape: [Time_horizon,H,W,C]
        # Time_horizon is fleble (can change during training)
        # print('change !')

        assert not check_nan(x)
        assert not check_inf(x)

        x = x.permute(0,3,1,2)   # [N,H,W,C] -> [N,C,H,W]
        x = self.norm_in(x)
        x = x.permute(0,2,3,1)   # [N,C,H,W] -> [N,H,W,C]

        x = self.in_proj(x)

        assert not check_nan(x)
        assert not check_inf(x)
        
        if self.has_PE:
            x = x + self.PE
        
        assert not check_nan(x)
        x = x.permute(0,3,1,2)   # [N,H,W,C] -> [N,C,H,W]

        assert not check_nan(x)
        assert not check_inf(x)

        for l in range(self.num_layers):
            if l%self.args.res_block_thickness==0:
                h = x
            x = self.layers[l](x)
            assert not check_nan(x), f'{l}'
            if (l+1)%self.args.res_block_thickness==0:
                x = x + h

        x = x.permute(0,2,3,1)   # [N,C,H,W] -> [N,H,W,C]

        x = self.out_proj(x)
        assert not check_nan(x)
        assert not check_inf(x)

        return x


class GCTLayer(nn.Module):
    def __init__(self, args):
        # mask shape: [H, W]

        super().__init__()

        self.args = args
        self.K = args.spatial_kernel_side_length
        self.num_head = args.num_head
        self.dim_head = args.dim_head

        self.spatial_kernel_size = [args.spatial_kernel_side_length, args.spatial_kernel_side_length]
        self.unfolder = nn.Unfold(kernel_size=self.spatial_kernel_size, padding=args.spatial_kernel_side_length//2, stride=1)

        self.init_kernel()

        return


    def update_cell_neighbor(self, mask):
        # This function CAN be called during training, and SHOULD be called when switching dataset (different dataset is assumed to have different mask).
        # mask shape: [S1, S2]
        # self.spatial_neighbor_mask shape: [S1,S2,  K1,K2], 1 means has neighbor, 0 means no neighbor
        S1,S2 = mask.shape
        self.spatial_neighbor_mask = torch.tensor(np.zeros([S1,S2,self.K,self.K])).bool()
        for s1 in range(S1):
            for s2 in range(S2):
                for k1 in range(self.K):
                    loc1 = s1 + k1 - self.K//2
                    if loc1<0 or loc1>=S1:
                        continue
                    for k2 in range(self.K):
                        loc2 = s2 + k2 - self.K//2
                        if loc2<0 or loc2>=S2:
                            continue
                        if mask[loc1,loc2]:
                            self.spatial_neighbor_mask[s1,s2,k1,k2] = True
        return

    def init_kernel(self):
        # self.kernels_k/v:   [Time_kernel,K1,K2]
        # self.kernels_q:    only one.

        Time_kernel = self.args.Time_kernel
        K = self.K

        self.kernels_q = nn.Linear(self.args.dim_hidden, self.args.dim_hidden)
        neurons = self.args.kernel_neurons

        k0 = []
        v0 = []
        for t_ in range(Time_kernel):
            k1 = []
            v1 = []
            for k1_ in range(K):
                k2 = []
                v2 = []
                for k2_ in range(K):
                    k2.append(getMLP(neurons))
                    v2.append(getMLP(neurons))
                k1.append(nn.ModuleList(k2))
                v1.append(nn.ModuleList(v2))
            k0.append(nn.ModuleList(k1))
            v0.append(nn.ModuleList(v1))
        self.kernels_k = nn.ModuleList(k0)
        self.kernels_v = nn.ModuleList(v0)

        return


    def forward(self, x):
        # maximum memory = input_size * K1*K2*Time_kernel*3(qkv) (maybe only a view, but true size is smaller)
        # input shape:  [N,C,H,W], C=dim_hidden, N=num of time slices
        # output shape: [N,C,H,W]
        
        # self.kernels_k/v:   [Time_kernel,K1,K2]
        # self.kernels_q:    only one.


        N,C,S1,S2 = x.shape
        Time_kernel = self.args.Time_kernel
        K = self.args.spatial_kernel_side_length

        xq = x.permute(0,2,3,1)  # [N,C,S1,S2] -> [N,S1,S2,C] 
        collect_q = self.kernels_q(xq)       # [N,S1,S2,C] 

        assert not check_nan(xq)


        u = unfold_NCS_2_NLKC(x, self.unfolder, self.spatial_kernel_size)  # [N, C, S1, S2] -> [N, L, K1, K2, C], L=S1*S2; preserve shape.
        
        assert not check_nan(u)

        u = unfold_time(u, Time_kernel)            # [N, L,K1,K2,C] -> [Time_kernel, N,  L,K1,K2,C]
        unfold_all = u.permute(0,3,4,1,2,5)    # [Time_kernel,K1,K2,  N,L,  C]
        res0_k = []
        res0_v = []
        for t in range(Time_kernel):
            # print('here, t=?',t)
            res1_k = []
            res1_v = []
            for k1 in range(K):
                res2_k = []
                res2_v = []
                for k2 in range(K):
                    kernel_k = self.kernels_k[t][k1][k2]
                    kernel_v = self.kernels_v[t][k1][k2]
                    z = unfold_all[t,k1,k2]     # shape = [N,L, C]

                    assert not check_nan(z)

                    ks = kernel_k(z)   #  [N,L, C]
                    vs = kernel_v(z)   #  [N,L, C]
                    assert not check_nan(ks)
                    assert not check_nan(vs)

                    res2_k.append(ks)
                    res2_v.append(vs)
                res1_k.append(torch.stack(res2_k))
                res1_v.append(torch.stack(res2_v))
            res0_k.append(torch.stack(res1_k))
            res0_v.append(torch.stack(res1_v))
        res0_k = torch.stack(res0_k).reshape(Time_kernel,K,K,  N,S1,S2,C)
        res0_v = torch.stack(res0_v).reshape(Time_kernel,K,K,  N,S1,S2,C)

        assert not check_nan(res0_k)
        assert not check_nan(res0_v)


        collect_k = res0_k.permute(3,4,5,  0,1,2,  6)  # [N,S1,S2, Time_kernel,K1,K2,  C]
        collect_v = res0_v.permute(3,4,5,  0,1,2,  6)  # [N,S1,S2, Time_kernel,K1,K2,  C]
        x = self.batch_aggregation(collect_q, collect_k, collect_v) # [N,C,S1,S2]

        assert not check_nan(x)

        return x

    def batch_aggregation(self, collect_q, collect_k, collect_v):
        # return shape:  [N,C,S1,S2]
        # collect_k/v: [N,S1,S2, Time_kernel,K1,K2,  C]
        # collect_q: [N,S1,S2,  C]
        # spatial_neighbor_mask:  [S1,S2,  K1,K2]
        N,S1,S2, Time_kernel,K1,K2,  C = collect_k.shape
        res0 = []
        for s1 in range(S1):
            res1 = []
            for s2 in range(S2):
                qs = collect_q[:,s1,s2,:]  # [N, C]
                ks = collect_k[:,s1,s2,...].view(N, Time_kernel*K1*K2, C)
                vs = collect_v[:,s1,s2,...].view(N, Time_kernel*K1*K2, C)

                neib_mask = self.spatial_neighbor_mask[s1,s2].reshape(-1).unsqueeze(0).expand(Time_kernel,-1).reshape(-1)

                y = self.once_neighbor_aggregation(qs,ks,vs,neib_mask)  # [N, C]
                res1.append(y)
            res1 = torch.stack(res1)  # [S2,N,C]
            res0.append(res1)
        res0 = torch.stack(res0)  # [S1,S2,N,C]
        res0 = res0.permute(2,3,0,1)
        return res0


    def once_neighbor_aggregation(self, qs,ks,vs,neib_mask):
        # qs:        [N,C]
        # ks,vs:     [N,num_neighbor,C]
        # neib_mask: [num_neighbo]
        # return shape: [N, C]
        N, _, C = ks.shape

        ks = ks[:,neib_mask,:].view(N, -1, self.num_head, self.dim_head).permute(0,2,1,3)
        vs = vs[:,neib_mask,:].view(N, -1, self.num_head, self.dim_head).permute(0,2,1,3)
        # k/v shape: [N, num_head, num_neighbor, dim_head]

        qs = qs.view(N,self.num_head,self.dim_head).unsqueeze(-1)
        # q shape:   [N, num_head, dim_head, 1]

        kq = torch.matmul(ks,qs).transpose(-1,-2)   # shape: [N, num_head, 1, num_neighbor]
        kq = F.softmax(kq, dim=-1)
        out = torch.matmul(kq, vs)            # shape: [N, num_head, 1, dim_head]
        out = out.squeeze(2).reshape(N, C)      # shape: [N, num_head*dim_head]
        return out




def unfold_time(x, Time_kernel):
    # This function assume Time_kernel is an odd number.
    # input shape:      [N_time, ...]   ('...'=rsh)
    # output shape:     [Time_kernel, N_time, ...]
    assert Time_kernel//2!=Time_kernel/2, 'Time_kernel should be an odd number, otherwise re-implement this function.'
    N = len(x)
    rsh = x.shape[1:]
    L1 = Time_kernel//2  # 5 -> L1=2 ; t-2, t-1, t, t+1, t+2
    us = []
    for t in range(N):
        if t<L1:
            elem = torch.cat([torch.zeros(L1-t, *rsh, dtype=x.dtype).to(x.device), x[0:t+L1+1, ...]], dim=0) # shape: [Time_kernel, *rsh]
            us.append(elem)
        elif t+L1+1>N:           # t=012...9 ; t+2+1>10, t>7, t=8,9
            elem = torch.cat([x[t-L1:N , ...], torch.zeros(t+L1+1-N, *rsh, dtype=x.dtype).to(x.device)], dim=0) # shape: [Time_kernel, *rsh]
            us.append(elem)
        else:
            us.append(x[t-L1:t+L1+1,...])
    u = torch.stack(us)  # [N, Time_kernel, ...]
    u = u.transpose(0,1) # [Time_kernel, N, ...]
    return u




def unfold_NCS_2_NLKC(arr_NCS, unfolder, kernel_size):
    # this function 'unfold' the input data.
    # input dim (for image): [N, C, S1, S2]
    # output dim(for image): [N, L, K1, K2, C]
    # N: batch; is uni-dimension.
    # C: channel;  is uni-dimension.
    # S: spacial dimension, or, [H,W] of image; is 2D or 3D or higher dimensional, depending on the data; for image, it is 2D.
    # L: number of unfolded kernel groups; is uni-dimension.
    # K: kernel size; is multi-dimension, for image it is 2D.

    # this function is applicable to kernel size (data shape) of 2D, 3D, ... any shape.
    # in returned shape, K is expanded
    arr__N_CK_L = unfolder(arr_NCS)
    N, CK, L = arr__N_CK_L.shape
    C, K = CK//np.prod(kernel_size), np.prod(kernel_size)
    arr__N_L_C_K = arr__N_CK_L.permute(0,2,1).view(N,L,C,K)
    arr__NLKC = arr__N_L_C_K.permute(0,1,3,2)
    Ks = list(kernel_size)
    shape_ = [N,L] + Ks + [C]
    arr__NLKC_KExpand = arr__NLKC.view(*shape_)
    return arr__NLKC_KExpand
    

def demo_funfold_preserve_image_shape():
    # preserve shape: means the output has the same shape as the input; should:
        # 1. kernel_side_length is odd number
        # 2. padding=kernel_side_length//2
        # 3. stride=1
    kernel_side_length = 3
    kernel_size=(kernel_side_length, kernel_side_length)
    
    unfolder = nn.Unfold(kernel_size=kernel_size, padding=kernel_side_length//2, stride=1)
    x = torch.arange(40).reshape(1, 2, 4, 5).float()
    
    y = unfold_NCS_2_NLKC(x, unfolder,kernel_size)
    
    p('x.shape:  [N, C, S1, S2]',x.shape,'\ny.shape: [N, L, K1, K2, C]',y.shape)
    print('-'*20)
    
    print(x)
    
    print(' --- visualize each channel ---')
    for c in range(y.shape[-1]):
        print(y[... ,c])
        print('-'*8)
    print('-'*12)
