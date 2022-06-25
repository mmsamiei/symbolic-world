import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import math


class ModLin(nn.Module):

    def __init__(self, code_vec, W_mat, b_vec):
        '''
        code_vec = tensor [size_cond]
        W_mat = tensor [size_out, size_in]
        b_vec = tensor [size_out]
        '''
        super(ModLin, self).__init__()
        self.code_vec = code_vec
        self.W_mat = W_mat
        self.b_vec = b_vec
        self.w_c_fc = nn.Linear(self.code_vec.shape[-1], self.W_mat.shape[-1])
        self.layernorm1 = nn.LayerNorm(self.W_mat.shape[-1])

    def forward(self, x):
        temp = x
        transformed_code = self.layernorm1(self.w_c_fc(self.code_vec))
        temp = transformed_code * temp
        temp = temp @ self.W_mat.T
        temp = temp + self.b_vec
        return temp

class ModMLP(nn.Module):

    def __init__(self, code_vec, W_mat, b_vec, num_layers=2):
        '''
        code_vec = tensor [size_cond]
        W_mat = tensor [size_out, size_in]
        b_vec = tensor [size_out]
        '''
        super(ModMLP, self).__init__()
        self.code_vec = code_vec
        self.W_mat = W_mat
        self.b_vec = b_vec
        self.w_c_fc = nn.Linear(self.code_vec.shape[-1], self.W_mat.shape[-1])
        self.layernorm1 = nn.LayerNorm(self.W_mat.shape[-1])
        self.mod_lin = ModLin(self.code_vec, self.W_mat, self.b_vec)
        self.num_layers = num_layers
        layers = []
        for i in range(self.num_layers):
            layers.append(ModLin(self.code_vec, self.W_mat, self.b_vec))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ModMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, code, W, b, dropout):
        super(ModMultiHeadAttentionLayer, self).__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = W.shape[0] // n_heads
        self.code = code
        self.W = W
        self.b = b
        
        self.fc_q = ModLin(self.code, self.W, self.b)
        self.fc_k = ModLin(self.code, self.W, self.b)
        self.fc_v = ModLin(self.code, self.W, self.b)
        
        self.fc_o = ModLin(self.code, self.W, self.b)
        
        self.dropout = nn.Dropout(dropout)
        
        #self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, compability_u, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        #compability_u = [batch, value len]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        #energy = torch.matmul(Q, K.permute(0, 1, 3, 2))  / self.scale ###TODO

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        

        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)    
        
        #attention = [batch size, n heads, query len, key len]

        c_u_i = compability_u.repeat(self.n_heads, attention.shape[2],1,1).permute(2,0,3,1)
        ### [head, len, batch, len] -> [batch, head, len, len]
        c_u_j = c_u_i.permute(0,1,3,2)


        attention = c_u_i * c_u_j * attention
        attention = attention / (1e-6 + attention)
        
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class LOC(nn.Module):

    def __init__(self, hid_dim, n_heads, code, W, b, num_mlp_layers, dropout):
        super(LOC, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.code = code
        self.W = W
        self.b = b
        self.dropout = dropout
        
        self.layernorm1 = nn.LayerNorm(hid_dim)
        self.layernorm2 = nn.LayerNorm(hid_dim)
        self.mod_attn = ModMultiHeadAttentionLayer(hid_dim, n_heads, code, W, b, dropout)
        self.mod_mlp = ModMLP(code, W, b, num_layers=num_mlp_layers)
    
    def forward(self, x, c_u):

        #x = [batch size, len, hid dim]
        #c_u = [batch, len]
        #value = [batch size, value len, hid dim]
        #compability_u = [value len]
        
        temp = x 
        temp = self.layernorm1(temp)
        a_u_hat = self.mod_attn(temp,temp,temp,c_u)[0]
        # a_u_hat = [batch, len, hid]
        a_u = a_u_hat * c_u.unsqueeze(-1).repeat_interleave(repeats=a_u_hat.shape[2], dim=-1) + x
        b_hat_u = self.mod_mlp(self.layernorm2(a_u))
        y_u = b_hat_u * c_u.unsqueeze(-1).repeat_interleave(repeats=a_u_hat.shape[2], dim=-1) + a_u
        return y_u

    

class DiniFunc(nn.Module):

    def __init__(self, hid_dim, n_heads, type_size, code_size, num_locs, W, b, num_mlp_layers, dropout):
        super(DiniFunc, self).__init__()
        self.type_vec = nn.parameter.Parameter(F.normalize(torch.randn(type_size), dim=0), requires_grad=True)
        self.code_vec = nn.parameter.Parameter(F.normalize(torch.randn(code_size), dim=0), requires_grad=True)
        loc_factory_kwargs = {'hid_dim':hid_dim, 'n_heads':n_heads , 'code':self.code_vec,\
                                'W': W, 'b': b, 'num_mlp_layers': num_mlp_layers, 'dropout': dropout}
        self.locs = nn.ModuleList([LOC(**loc_factory_kwargs) for i in range(num_locs)])
        self.layernorm1 = nn.LayerNorm(hid_dim)

    
    def get_compatibility_score(self, type_x):
        ### type_x = tensor [batch, len, type_size]
        compatibility = type_x @ self.type_vec
        ### [batch,len]
        return compatibility
    
    def set_type(self, type_vec):
        self.type_vec.data = F.normalize(type_vec, dim=0)

    def forward(self, x, c_u):
        temp = x 
        for loc in self.locs:
            temp = temp + loc(temp, c_u)
        temp = self.layernorm1(temp)
        return temp


class DiniFuncRow(nn.Module):

    def __init__(self, hid_dim, n_heads, num_funcs, type_size, code_size, type_inference_depth, type_inference_width, threshold, num_locs, num_mlp_layers, dropout):
        super(DiniFuncRow, self).__init__()
        self.W = nn.parameter.Parameter(torch.randn(hid_dim, hid_dim), requires_grad=True)
        self.b =nn.parameter.Parameter(torch.randn(hid_dim), requires_grad=True)
        self.threshold = threshold
        self.type_inference_depth = type_inference_depth
        self.type_inference_width = type_inference_width
        func_factory_kwargs = {'hid_dim': hid_dim, 'n_heads': n_heads, 'type_size':type_size,\
        'code_size':code_size, 'num_locs': num_locs, 'W':self.W, 'b':self.b, 'num_mlp_layers':num_mlp_layers,\
        'dropout': dropout}

        funcs_types = torch.randn((num_funcs, type_size))
        nn.init.orthogonal_(funcs_types)

        self.funcs = nn.ModuleList([DiniFunc(**func_factory_kwargs) for i in range(num_funcs)])
        for i, func in enumerate(self.funcs):
            func.set_type(funcs_types[i])
            
        #self.type_inference = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, type_size))
        
        
        type_inference_list = [nn.Linear(hid_dim, type_inference_width), nn.ReLU()] +\
            [sublayer for i in range(type_inference_depth) for sublayer in [nn.Linear(type_inference_width, type_inference_width), nn.ReLU()]] +\
            [nn.Linear(type_inference_width, type_size)]

        self.type_inference = nn.Sequential(*type_inference_list)
        self.sigma = 1
        self.layernorm1 = nn.LayerNorm(hid_dim)

    def get_compability_matrix(self, x):
        compability_list = []
        types_x = self.type_inference(x) ## [batch, len, type_size]
        for func in self.funcs:
            compability = func.get_compatibility_score(types_x)
            compability_list.append(compability)
        compability_matrix = torch.stack(compability_list, dim=-1)
        temp = compability_matrix
        compability_matrix = torch.exp(-1*(1-temp)/self.sigma) * (compability_matrix > self.threshold)
        compability_matrix = compability_matrix / (compability_matrix.sum(dim=2).unsqueeze(-1).repeat_interleave(repeats=compability_matrix.shape[2], dim=-1) + 1e-6)
        return compability_matrix

    def forward(self, x):
        # x = [batch, len, size]
        temp = x
        compability_matrix = self.get_compability_matrix(x)
        for i_func,func in enumerate(self.funcs):
            temp = temp + func(x, compability_matrix[:,:,i_func]) * compability_matrix[:,:,i_func].unsqueeze(-1).repeat_interleave(repeats=temp.shape[2], dim=-1)
        temp = self.layernorm1(temp)
        return temp
    
      

class DiniFuncRowIter(nn.Module):

    def __init__(self, hid_dim, n_heads, num_iter, num_funcs, type_size, code_size, type_inference_depth, type_inference_width, threshold, num_locs, num_mlp_layers, dropout):
        super(DiniFuncRowIter, self).__init__()
        funcrow_factory_kwargs = {'hid_dim': hid_dim, 'n_heads': n_heads, 'num_funcs':num_funcs,\
        'type_size':type_size,'code_size':code_size, 'type_inference_depth': type_inference_depth,\
        'type_inference_width':type_inference_width, 'threshold':threshold, 'num_locs': num_locs,\
        'num_mlp_layers':num_mlp_layers, 'dropout': dropout}
        self.funcs_row = DiniFuncRow(**funcrow_factory_kwargs)
        self.num_iter = num_iter
        self.layernorm1 = nn.LayerNorm(hid_dim)
  

    def forward(self, x):
        # x = [batch, len, size]
        temp = x 
        for iter_num in range(self.num_iter):
            temp = temp + self.funcs_row(temp)
        temp = self.layernorm1(temp)
        return temp



class DiniEncoder(nn.Module):
    def __init__(self, hid_dim, n_heads, num_layer, num_iter, num_funcs, type_size, code_size, type_inference_depth, type_inference_width, threshold, num_locs, num_mlp_layers, dropout):
        super(DiniEncoder, self).__init__()
        
        diniFuncRowIter_factory_kwargs = {'hid_dim': hid_dim, 'n_heads': n_heads, 'num_iter':num_iter,\
        'num_funcs':num_funcs,'type_size':type_size,'code_size':code_size, 'type_inference_depth':type_inference_depth,\
        'type_inference_width':type_inference_width, 'threshold':threshold,'num_locs': num_locs,\
        'num_mlp_layers':num_mlp_layers, 'dropout': dropout}

        self.dim_model = hid_dim
        self.layers = nn.ModuleList([DiniFuncRowIter(**diniFuncRowIter_factory_kwargs) for i in range(num_layer)])
        self.layernorm1 = nn.LayerNorm(hid_dim)

    def forward(self, x):
        temp = x
        for layer in self.layers:
            temp = temp + layer(temp)
        temp = self.layernorm1(temp)
        return temp