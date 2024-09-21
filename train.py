import argparse
import torch
import torch.optim as optim

from model import AHFAN
from copy import deepcopy
from config import *
import pickle
import numpy as np
from sklearn.utils import shuffle
from pygod.utils import load_data

# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import manifold
import scipy.io
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
# from tensorboardX import SummaryWriter
# writer = SummaryWriter(log_dir='logs')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(data, model, criterion, optimizer, label, beta=.6):
    anomaly, normal = label
    idx_train = data.train_mask
    model.train()
    optimizer.zero_grad()
    output ,bias_loss = model(data.x, data.edge_index, label=(data.train_mask & anomaly, data.train_mask & normal))
    loss_train = criterion(output[idx_train], data.y[idx_train].long()) + bias_loss * beta
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def sample_mask(idx, l):
        """Create mask."""
        mask = torch.zeros(l)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)
    
# def dataset():
# # 读取.mat文件
#     #  data = scipy.io.loadmat('/home/hdou/model/first/AHFAN/AHFAN-main/dataset/Amazon.mat')
#      data = scipy.io.loadmat('first/MC-AGCN/AHFAN-main/dataset/YelpChi.mat')
#      edge_all = from_scipy_sparse_matrix(data['homo'])
#      edge_weight = edge_all[1]
#      edge_index = edge_all[0]
#      # 获取label值和features
#      label = torch.from_numpy(data['label'].astype(float)).long()

#      features = torch.from_numpy(data['features'].toarray().astype(float))
#      label = torch.squeeze(label)
#      # 创建Data对象
#      data_obj = Data(x=features, y=label,edge_index=edge_index)
#      return data_obj

# def dataset():
#      data = np.load('first/MC-AGCN/AHFAN-main/dataset/tolokers.npz')
#     #  data = np.load('second/sprectal-transformer/dataset/tolokers.npz')
#      node_features = torch.tensor(data['node_features'])
#      labels = torch.tensor(data['node_labels'])
#      edges = torch.tensor(data['edges'])
#     #  data = pickle.load(open('second/sprectal-transformer/dataset/yelp.dat', 'rb'))
#      # data = pickle.load(open('second/sprectal-transformer/dataset/elliptic.dat', 'rb'))
#      # data = load_data('reddit')
#      # data = load_data('Amazon')

#      data_obj = Data(x=node_features, y=labels, edge_index=edges.T)
#      # data_obj = Data(x=data.x, y=data.y, edge_index=data.edge_index,adj=adj)
#      # data_obj = Data(x=data.x, y=data.y, edge_index=data.edge_index, train_mask=data.train_mask ,
#      #                 val_mask= data.val_mask,test_mask= data.test_mask,adj=adj)
#      return data_obj


def main(args, exp_num=0):
    # data = dataset().to(device)
    
    # data = pickle.load(open('AHFAN-main/dataset/elliptic.dat'.format(args.dataset), 'rb'))
    # # # # data = Data(x=data1.x, y=data1.y,edge_index=data1.edge_index)
    # data = data.to(device)
    # # print('train_mask',sum(data.train_mask),'val_mask',sum(data.val_mask),'test_mask',sum(data.test_mask))
    # # print(sum(data.train_mask))
    # # data = load_data('reddit').to(device)    
    data = load_data('weibo').to(device)    
    # data = torch.load('first/MC-AGCN/AHFAN-main/dataset/weibo.pt').to(device)  
    sample_number = len(data.y) 
    seed = 42
    shuffled_idx = shuffle(np.array(range(len(data.y))), random_state=seed) 
    # train_idx = shuffled_idx[:int(0.6* data.y.shape[0])].tolist()
    # val_idx = shuffled_idx[int(0.6*data.y.shape[0]):int(0.9*data.y.shape[0])].tolist()
    # test_idx = shuffled_idx[int(0.9*data.y.shape[0]):].tolist()
    train_idx = shuffled_idx[int(0.3*data.y.shape[0]):int(0.9* data.y.shape[0])].tolist()
    val_idx = shuffled_idx[:int(0.3*data.y.shape[0])].tolist()
    test_idx = shuffled_idx[int(0.9*data.y.shape[0]):].tolist()
    train_mask = sample_mask(train_idx, sample_number)
    val_mask = sample_mask(val_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    data.train_mask = train_mask.to(device)
    data.test_mask = test_mask.to(device)
    data.val_mask = val_mask.to(device)
    
    
    net = AHFAN(in_channels=data.x.shape[1],  hid_channels=params_config['hidden_channels'],num_class=2)
    
    net.to(device)
 
    optimizer = optim.Adam([
        # dict(params=net.filters.parameters(), lr=params_config['lr_f']),     
        dict(params=net.filters1.parameters(), lr=params_config['lr_f']), 
        dict(params=net.lin, lr=params_config['lr'], weight_decay=params_config['weight_decay']),
        # dict(params=net.attn, lr=params_config['lr'], weight_decay=params_config['weight_decay'])
        ])

    weights = torch.Tensor([1., 1.])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    anomaly = (data.y == 1)
    normal = (data.y == 0)
    label = (anomaly, normal)
    
    c = 0
    auc_pr_best = 0
    auc_roc_best = 0
    auc_roc_test_epoch = 0
    auc_pr_test_epoch = 0
    best_net = None
   
    for epoch in range(params_config['epochs']):
        loss= train(data, net, criterion, optimizer, label, beta=params_config['beta'])
        auc_roc_val, auc_pr_val  = net.evaluating(data.x, data.y, data.edge_index, data.val_mask)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
            print('Epoch:{:04d}\tloss:{:.4f}\tVal AUC-ROC:{:.4f}\tVal AUC-PR:{:.4f}'
                    '\ttest AUC-ROC:{:.4f}\ttest AUC-PR:{:.4f}'
                    .format(epoch + 1, loss, auc_roc_val, auc_pr_val, auc_roc_test_epoch, auc_pr_test_epoch))
            
        if auc_pr_val >= auc_pr_best:
            auc_pr_best = auc_pr_val
            auc_roc_best = auc_roc_val
            auc_roc_test_epoch, auc_pr_test_epoch = net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
            best_net = deepcopy(net)
            c = 0
        else:
            c += 1
        if c == params_config['patience']:
            break

    auc_roc_test_exp, auc_pr_test_exp = best_net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
   
    return auc_roc_test_exp, auc_pr_test_exp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='weibo', help='Dataset [yelp, elliptic, weibo, quest]')
    parser.add_argument('--exp_num', type=int, default=20, help='Default Experiment Number')
    parser.add_argument('--eval_interval', type=int, default=100)
    args = parser.parse_args(args=[])
    params_config = dataset_config[args.dataset]
    auc_roc_list = []
    auc_pr_list = []

    for i in range(args.exp_num):
        auc_roc_test, auc_pr_test = main(args, exp_num=i)
        auc_roc_list.append(auc_roc_test)
        auc_pr_list.append(auc_pr_test)
    auc_df = pd.DataFrame(auc_roc_list, columns=['AUROC'])
    pre_df = pd.DataFrame(auc_pr_list, columns=['AUPRC'])
    
    auc_filename = f"auc_trial.xlsx"
    pre_filename = f"pre_trial.xlsx"

    auc_df.to_excel(auc_filename, index=False)
    pre_df.to_excel(pre_filename, index=False)
        
    print("AUC ROC Mean:{:.5f}\tStd:{:.5f}\tAUC PR Mean:{:.5f}\tStd:{:.5f}".format(np.mean(auc_roc_list),
                                                                                   np.std(auc_roc_list),
                                                                                   np.mean(auc_pr_list),
                                                                                   np.std(auc_pr_list)))
