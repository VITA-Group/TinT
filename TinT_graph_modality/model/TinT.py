
from .model_utils import *



class LocAwareKernel(nn.Module):
    def __init__(self, args, use_niso_node_ker):
        super().__init__()

        self.args = args
        self.adjs = args.adjs
        self.self_graph_attn = GCN_FFN(args, args.norm_Adj_matrix)
        self.use_niso_node_ker = use_niso_node_ker
        if use_niso_node_ker:
            # from torch_geometric.nn import TransformerConv
            # self.loc_kernels = nn.ModuleList([TransformerConv(args.d_model, args.d_model, edge_dim=0, beta=True) for i in range(len(self.adjs))])
            self.loc_kernels = nn.ModuleList([GraphAttn(args.d_model, args.d_model) for i in range(len(args.adjs))])
        return

    def forward(self, query, key, value, mask=None):
        if self.use_niso_node_ker:
            return self.self_graph_attn(query) + self.forward_niso(query, key, value, mask)
        else:
            return self.self_graph_attn(query)


    def forward_niso(self, query, key, value, mask=None):
        # encoder: q=k=v=x
        # decoder: q,k,v = x, m, m;  gcn_ffn use x.
        # inputs/output: [B,N,T,dim]
        _B, _N, _T, _C = value.shape
        query = query.view(_B, -1, _C)
        value = value.view(_B, -1, _C)

        context_layer = []
        for vdata, qdata in zip(value, query): # value: [B,N,dim]
            gout_Nd = []
            for i in range(len(self.adjs)):
                gout_Nd.append(self.loc_kernels[i]((vdata, qdata), self.adjs[i]))
                # gout_Nd.append(self.loc_kernels[i]((vdata, qdata), self.adjs[i].indices(), self.adjs[i].values()))
                # OOM?

            gout_Nd = torch.sum(torch.stack(gout_Nd), dim=0)
            context_layer.append(gout_Nd)
        context_layer = torch.stack(context_layer).view(_B, _N, -1, _C)

        return context_layer




class EncoderSelfAttn(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, args):
        super().__init__()
        nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, dropout = args.nb_head, args.d_model, args.num_of_weeks, args.num_of_days, args.num_of_hours, args.num_for_predict, args.kernel_size, args.dropout
        self.args = args
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = nn.ModuleList([nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)) for _ in range(2)])  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour


        # ---- init for volume_kernel ----
        self.dim_head_volkernel = int(args.dim_hidden_A / args.num_head_volkernel)
        self.all_head_size = args.num_head_volkernel * self.dim_head_volkernel
        self.vol_query = nn.Linear(args.dim_hidden_A, self.all_head_size)
        self.vol_key = nn.Linear(args.dim_hidden_A, self.all_head_size)
        self.vol_value = nn.Linear(args.dim_hidden_A, self.all_head_size)

        # ---- init for locaware_kernel ----
        self.adjs = args.adjs
        
        if self.args.enc_has_node:
            self.locaware_kernel = LocAwareKernel(args, args.enc_use_niso_node_ker)

    def forward(self, query, key, value, tokenizer, mask=None, query_multi_segment=False, key_multi_segment=False):
        res = self.time_kernel(query, key, value, mask, query_multi_segment, key_multi_segment)
        if self.args.enc_has_node:
            res += self.locaware_kernel(query, key, value, mask)
        if self.args.enc_has_volume:
            res += self.volume_kernel(query, key, value, tokenizer, mask)
        return res

    def volume_kernel(self, query, key, value, tokenizer, mask=None):
        # inputs/output: [B,N,dim]
        _B, _N, _T, _C = query.shape
        query = query.view(_B, -1, _C)
        key = key.view(_B, -1, _C)
        value = value.view(_B, -1, _C)

        mixed_query_layer = self.vol_query(tokenizer(query, 'in'))   # [512, 197, 768] <- [512, 197, 768]
        mixed_key_layer = self.vol_key(tokenizer(key, 'in'))
        mixed_value_layer = self.vol_value(tokenizer(value, 'in'))

        query_layer = transpose_for_scores(mixed_query_layer, self.args.num_head_volkernel)  # [512, 12, 197, 64] <- [512, 197, 768] (N_head=12)
        key_layer = transpose_for_scores(mixed_key_layer, self.args.num_head_volkernel)
        value_layer = transpose_for_scores(mixed_value_layer, self.args.num_head_volkernel)  # [512, 12, 197, 64] <- [512, 197, 768]

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / np.sqrt(self.dim_head_volkernel)      # shape: [B, num_head, N, N]
        # attention_probs = self.softmax(attention_scores)   # [512, 12, 197, 197]
        # weights = attention_probs if self.vis else None
        # attention_probs = self.attn_dropout(attention_probs)
        # context_layer = torch.matmul(attention_probs, value_layer)  # BHNd = [512, 12, 197, 64]
        context_layer = attention(query_layer, key_layer, value_layer)[0]


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # [512, 197, 768]
        context_layer = tokenizer(context_layer, 'out').view(_B, _N, _T, _C)
        return context_layer


    def time_kernel(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length:self.w_length+self.d_length, :], key[:, :, self.w_length:self.w_length+self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)

        x = x.view(nbatches, N, -1, self.h * self.d_k)  # [B, N, T, C]


        return self.linears[-1](x)


class DecoderCrossAttn(nn.Module):  # query: causal conv; key 1d conv
    def __init__(self, args):
        super().__init__()
        self.args = args
        nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, dropout = args.nb_head, args.d_model, args.num_of_weeks, args.num_of_days, args.num_of_hours, args.num_for_predict, args.kernel_size, args.dropout
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1)//2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour
        if self.args.dec_has_node:
            self.locaware_kernel = LocAwareKernel(args, args.dec_use_niso_node_ker)


    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        res = self.time_kernel(query, key, value, mask, query_multi_segment, key_multi_segment)
        if self.args.dec_has_node:
            res += self.locaware_kernel(query, key, value, mask)
        if self.args.dec_has_volume:
            res += self.volume_kernel(query, key, value, tokenizer, mask)

        return res

    def time_kernel(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):


        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''
        

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w = self.query_conv1Ds_aware_temporal_context(query[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length+self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1,
                                                                                                                4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.key_conv1Ds_aware_temporal_context(key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)





        x = x.view(nbatches, N, -1, self.h * self.d_k)  # [B, N, T, C], same as query input


        return self.linears[-1](x)





class EncoderLayer(nn.Module):
    def __init__(self, args, residual_connection=True, use_LayerNorm=True):
        super().__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        size, dropout = args.d_model, args.dropout

        self.self_attn = EncoderSelfAttn(args)

        if args.ffn_use_MLP:
            self.ffn = getMLP([args.d_model, int(args.d_model*1.5), args.d_model])
        else:
            self.ffn = GCN_FFN(args, args.norm_Adj_matrix)

        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout, residual_connection, use_LayerNorm) for _ in range(2)])
        self.size = size

    def forward(self, x, tokenizer):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tokenizer, query_multi_segment=True, key_multi_segment=True))
        return self.sublayer[1](x, self.ffn)



class DecoderLayer(nn.Module):
    def __init__(self, args, residual_connection=True, use_LayerNorm=True):
        super().__init__()
        self.size = args.d_model

        self.self_attn = DecoderSelfAttn(args.nb_head, args.d_model, args.num_of_weeks, args.num_of_days, args.num_of_hours, args.num_for_predict, args.kernel_size, dropout=args.dropout)  # decoder的trend-aware attention用因果卷积
        self.cross_attn = DecoderCrossAttn(args)
        
        if args.ffn_use_MLP:
            self.ffn = getMLP([args.d_model, int(args.d_model*1.5), args.d_model])
        else:
            self.ffn = GCN_FFN(args, args.norm_Adj_matrix)
        
        self.sublayer = nn.ModuleList([SublayerConnection(args.d_model, args.dropout, residual_connection, use_LayerNorm) for _ in range(3)])

    def forward(self, x, memory):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False))  # output: (batch, N, T', d_model)
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))  # output: (batch, N, T', d_model)
        return self.sublayer[2](x, self.ffn)  # output:  (batch, N, T', d_model)


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.num_layers)])
        self.norm = nn.LayerNorm(self.layers[0].size)


    def forward(self, x, tokenizer):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x = layer(x, tokenizer)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()        
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.num_layers)])
        self.norm = nn.LayerNorm(self.layers[0].size)

    def forward(self, x, memory):
        '''
        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)

def make_model(args_out, DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True):

    # LR rate means: graph Laplacian Regularization

    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵

    num_of_vertices = norm_Adj_matrix.shape[0]


    # encoder temporal position embedding
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)

    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7*24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index


    args = get_default_args(args_out, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout, norm_Adj_matrix, num_layers, decoder_output_size)


    encoder_embedding = nn.Sequential(nn.Linear(encoder_input_size, d_model), TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index), SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN_PE(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num))
    decoder_embedding = nn.Sequential(nn.Linear(decoder_output_size, d_model), TemporalPositionalEncoding(d_model, dropout, num_for_predict), SpatialPositionalEncoding(d_model, num_of_vertices, dropout, GCN_PE(norm_Adj_matrix, d_model, d_model), smooth_layer_num=smooth_layer_num))

    encoder = Encoder(args)

    decoder = Decoder(args)

    model = EncoderDecoder(args,
                           encoder,
                           decoder,
                           encoder_embedding,
                           decoder_embedding,
                           DEVICE)
    # param init
    for p in model.parameters():
        try:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        except ValueError:
            pass


    return model


class D: pass


def get_default_args(args_out, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict, kernel_size, dropout, norm_Adj_matrix, num_layers, decoder_output_size):

    args = D()
    args.args_out = args_out
    args.adjs = build_non_iso_adjs(args_out)
    args.desired_finer_token_dim        =           4

    args.device                         =           torch.device(f'cuda:{int(args_out["Training"]["cudaID"])}')
    args.num_feat                       =           int(args_out['Training']['in_channels'])
    args.has_PE                         =           False
    args.vis                            =           False

    args.dim_constr                     =           5
    args.dim_var                        =           17
    args.graph_reg_type                 =           [ 'force sparse support',
                                                    'v0_downgrade_gnn_no_softmax', 
                                                    'v1_gnn_plus_normal_no_softmax',
                                                    ][1]

    args.has_cls_token                  =           False
    args.input_embedding_dropout        =           0.05
    args.dim_out                        =           1


    args.transformer_depth              =           2

    args.ffn_dropout_rate               =           0.1

    args.outer_ffn_dims                 =           []
    args.ffn_normfun                    =           ['', 'layernorm' , 'batchnorm'][1]
    args.use_kq                         =           0

    args.num_head_volkernel               =           2



    # ----- init dataset dependent args -----
    dataset = args_out['Data']['dataset_name']      # ['sd', 'la_RU', 'la_DT', 'sf']
    args.n_patches                  =           64
    if dataset=='sd':  # go to [def tokenizer] to find answer
        # args._x2 = 2
        # args.dim_token                  =           324
        # args.dim_hidden_A               =           args.dim_token * args._x2
        args.dim_hidden_A               =           648

    elif dataset=='la_RU':
        args.dim_hidden_A                =           524
    elif dataset=='la_DT':
        args.dim_hidden_A                =           1076
    elif dataset=='sf':
        args.dim_hidden_A                =           1020

    else:
        raise NotImplementedError


    # args.use_background                 = 1
    # args.num_kernel                     = 12
    args.num_reshape_token              = 64



    model_mode = ['orig_astgnn', 'gct', 'full','other'][2]
    if model_mode == 'orig_astgnn':
        # ==== fix below ====
        args.ffn_use_MLP        =   0
        args.enc_has_node       =   0
        args.enc_has_volume     =   0
        args.enc_use_niso_node_ker  =  0
        args.dec_has_node       =   0
        args.dec_use_niso_node_ker  =  0
        args.dec_has_volume     =   0
    elif model_mode == 'gct':
        # ==== fix below ====
        args.ffn_use_MLP        =   1
        args.enc_has_node       =   1
        args.enc_has_volume     =   0
        args.enc_use_niso_node_ker  =  1
        args.dec_has_node       =   1
        args.dec_use_niso_node_ker  =  0
        args.dec_has_volume     =   0

    elif model_mode == 'full':
        # ==== fix below ====
        args.ffn_use_MLP        =   1
        args.enc_has_node       =   1
        args.enc_has_volume     =   1
        args.enc_use_niso_node_ker  =  1
        args.dec_has_node       =   1
        args.dec_use_niso_node_ker  =  0
        args.dec_has_volume     =   0
        print('\n\n must be full... \n\n ')

    else:
        args.ffn_use_MLP        =   1
        args.enc_has_node       =   1
        args.enc_has_volume     =   1
        args.enc_use_niso_node_ker  =  1
        args.dec_has_node       =   1
        args.dec_use_niso_node_ker  =  0
        args.dec_has_volume     =   0



    args.nb_head            = nb_head
    args.d_model            = d_model
    args.num_of_weeks       = num_of_weeks
    args.num_of_days        = num_of_days
    args.num_of_hours       = num_of_hours
    args.num_for_predict    = num_for_predict
    args.kernel_size        = kernel_size
    args.dropout            = dropout
    args.norm_Adj_matrix    = norm_Adj_matrix
    args.num_layers         = num_layers
    args.decoder_output_size = decoder_output_size

    return args




