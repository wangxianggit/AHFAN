import os.path as osp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.):
        super(SAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels,hidden_channels))
        
        self.dropout = dropout
        
    def reset_parameters():
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index)
        
        return x.log_softmax(dim=-1)
