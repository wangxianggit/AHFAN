import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
from utils import aucPerformance
import dgl.function as fn
from torch_geometric.nn import AGNNConv,GCNConv,GATv2Conv
from multi_head import MultiHeadAttention
from gat import GATConv

class AHFAN(nn.Module):
    def __init__(self, in_channels, hid_channels, num_class, dropout=0.5):
        super(AHFAN, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()

        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels,hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels,hid_channels),
                                                 )
        
        self.filters = nn.ModuleList([GCNConv(hid_channels,hid_channels)  for _ in range(2)])
        self.filters2 = AGNNConv(hid_channels,hid_channels)
        # self.filters1 = GATConv(hid_channels,hid_channels)
        # self.filters1 = GATv2Conv(hid_channels,hid_channels)
        
        ### Multi-head
        # self.filters1 = nn.ModuleList([MultiHeadAttention(in_channels=in_channels, hid_channels=hid_channels, n_head=2)
        #                                for _ in range(2)])
        
        self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                 self.attn_fn,
                                 )
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_channels*2, num_class))

        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
     


    def forward(self, x, edge_index, label=None):
        x = self.linear_transform_in(x.to(torch.float32))
        # _x = x
        h_list = []
        for i, filter_ in enumerate(self.filters):
            h = filter_(x, edge_index) 
            h_list.append(h)
        h1  = self.filters2(x, edge_index) 
        
        ### Multi-head
        # for filters1 in self.filters1:
        #     x_new = self.filters2(x, edge_index)
        #     h1  = filters1(x_new, x_new, x) 
        #     H = h1 + _x
        # h1  = self.filters1(x, x, x) 
        # h_list.append(h1)
        
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1) 

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        self.score = soft_score
        
        res = h_filters[:,0,:] * soft_score[:, 0]
        for i in range(1, 2):
            res += h_filters[:, i, :] * soft_score[:, i]
        
        res = torch.cat((res,h1),dim=1)
        y_hat = self.linear_cls_out(res)
        marginal_loss = 0.

        if self.training:
            anomaly_train, normal_train = label
            normal_bias = soft_score[normal_train][:, 1] - soft_score[normal_train][:, 0]
            anomaly_bias = soft_score[anomaly_train][:, 0] - soft_score[anomaly_train][:, 1]
            normal_bias = torch.clamp(normal_bias, -0.)
            anomaly_bias = torch.clamp(anomaly_bias, -0.)
            normal_bias = torch.mean(normal_bias)
            anomaly_bias = torch.mean(anomaly_bias)
            bias = anomaly_bias + normal_bias
            marginal_loss = bias

        if self.training:
            return y_hat,  marginal_loss 
        else:
            return y_hat

    @torch.no_grad()
    def evaluating(self, x, y, edge_index, test_index):
        self.eval()
        y_pred = self.forward(x, edge_index)
        y_pred = F.softmax(y_pred, dim=1)[:, 1]
        self.train()
        y_test = y[test_index]
        y_pred = y_pred[test_index]
        y_test = y_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        auc_roc, auc_pr  = aucPerformance(y_test, y_pred)
        return auc_roc, auc_pr


    # @torch.no_grad()
    # def get_attn(self, label, train_index, test_index):
    #     anomaly, normal = label
    #     # test_attn_anomaly = list(chain(*torch.mean(self.score[test_index & anomaly], dim=0).tolist()))
    #     # test_attn_normal = list(chain(*torch.mean(self.score[test_index & normal], dim=0).tolist()))
    #     train_attn_anomaly = list(chain(*torch.mean(self.score[train_index & anomaly], dim=0).tolist()))
    #     a = train_attn_anomaly[0]
    #     b = train_attn_anomaly[1]
    #     # c = train_attn_anomaly[2]
    #     train_attn_normal = list(chain(*torch.mean(self.score[train_index & normal], dim=0).tolist()))
    #     d = train_attn_normal[0]
    #     e = train_attn_normal[1]
    #     # f = train_attn_normal[1]
    #     return a,b,d,e
    # def gdc(g, A_loop, alpha: float, eps: float):
        
        
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    #     N = g.num_nodes()
    #     # src, dst = g.edges()
    #     # # Convert to GPU tensors
    #     # src = src.to(device)
    #     # dst = dst.to(device)
    #     # # Create the adjacency matrix as a sparse tensor
    #     # indices = torch.stack([src, dst])
    #     # values = torch.ones(src.shape[0], device=device)
    #     # A = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    #     # # Add self-loops
        
    #     # I = torch.eye(N, device=device).to_sparse()
    #     # A_loop = A + I
        
        
    #     # Symmetric transition matrix
    #     D_loop_vec = torch.sparse.sum(A_loop, dim=1).to_dense()
    #     D_loop_vec_invsqrt = 1.0 / torch.sqrt(D_loop_vec)
    #     D_loop_invsqrt = torch.diag(D_loop_vec_invsqrt)
    #     T_sym = D_loop_invsqrt @ A_loop.to_dense() @ D_loop_invsqrt
    #     T_sym = T_sym.to_sparse()
        
    #     # PPR-based diffusion
    #     I = torch.eye(N, device=device).to_sparse()
    #     T_sym = (1 - alpha) * T_sym
    #     S = alpha * torch.inverse(I.to_dense() - T_sym.to_dense()).to_sparse()
        
    #     # Sparsify using threshold epsilon
    #     S = S.coalesce()
    #     mask = S.values() >= eps
    #     S_tilde_indices = S.indices()[:, mask]
    #     S_tilde_values = S.values()[mask]
    #     S_tilde = torch.sparse_coo_tensor(S_tilde_indices, S_tilde_values, S.shape).coalesce()
        
    #     # Column-normalized transition matrix on graph S_tilde
    #     D_tilde_vec = torch.sparse.sum(S_tilde, dim=1).to_dense()
    #     D_tilde_inv = 1.0 / D_tilde_vec
    #     D_tilde_inv[torch.isinf(D_tilde_inv)] = 0.0
    #     D_tilde_invsqrt = torch.diag(D_tilde_inv)
    #     T_S = (S_tilde.to_dense() @ D_tilde_invsqrt).to_sparse()
        
    #     # Extract COO format
    #     T_S = T_S.coalesce()
    #     new_src, new_dst = T_S.indices()
    #     new_weights = T_S.values()
        
    #     # Create the DGL graph
    #     # g_dgl = dgl.graph((new_src, new_dst), num_nodes=N)
    #     # g_dgl.edata['weight'] = new_weights
        
    #     return g_dgl
  
    

