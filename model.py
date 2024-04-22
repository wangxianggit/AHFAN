import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
from utils import aucPerformance
import dgl.function as fn
from torch_geometric.nn import AGNNConv,GCNConv,GATConv,GATv2Conv


class AMNet(nn.Module):
    def __init__(self, in_channels, hid_channels, num_class, dropout=0.5):
        super(AMNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()

        self.linear_transform_in = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                                 self.act_fn,
                                                 nn.Linear(hid_channels,hid_channels),
                                                #  self.act_fn,
                                                #  nn.Linear(hid_channels,hid_channels),
                                                 )
        
        # self.filters = nn.ModuleList([GCNConv(hid_channels,hid_channels)  for _ in range(2)])
        self.filters1 = AGNNConv(requires_grad = True)
        # self.filters1 = GATv2Conv(hid_channels,hid_channels)
        # self.W_f = nn.Sequential(nn.Linear(hid_channels, hid_channels),
        #                          self.attn_fn,
        #                          )
        # self.W_x = nn.Sequential(nn.Linear(hid_channels, hid_channels),
        #                          self.attn_fn,
        #                          )
        self.linear_cls_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_class))

        # self.attn = list(self.W_x.parameters())
        # self.attn.extend(list(self.W_f.parameters()))
        
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
     


    def forward(self, x, edge_index, label=None):
        x = self.linear_transform_in(x.to(torch.float32))
        h_list = []
        # for i, filter_ in enumerate(self.filters):
        #     h = filter_(x, edge_index) 
        #     h_list.append(h)
        h1  = self.filters1(x, edge_index) 
      
        # # h_list.append(h1)
        # h_filters = torch.stack(h_list, dim=1)
        # h_filters_proj = self.W_f(h_filters)
        # x_proj = self.W_x(x).unsqueeze(-1) 

        # score_logit = torch.bmm(h_filters_proj, x_proj)
        # soft_score = F.softmax(score_logit, dim=1)
        # self.score = soft_score
        
        # res = h_filters[:,0,:] * soft_score[:, 0]
        # for i in range(1, 2):
        #     res += h_filters[:, i, :] * soft_score[:, i]
        
        # res = torch.cat((res,h1,x),dim=1)
        y_hat = self.linear_cls_out(h1)
        marginal_loss = 0.

        # if self.training:
        #     anomaly_train, normal_train = label
        #     normal_bias = soft_score[normal_train][:, 1] - soft_score[normal_train][:, 0]
        #     anomaly_bias = soft_score[anomaly_train][:, 0] - soft_score[anomaly_train][:, 1]
        #     normal_bias = torch.clamp(normal_bias, -0.)
        #     anomaly_bias = torch.clamp(anomaly_bias, -0.)
        #     normal_bias = torch.mean(normal_bias)
        #     anomaly_bias = torch.mean(anomaly_bias)
        #     bias = anomaly_bias + normal_bias
        #     marginal_loss = bias

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

  
    

