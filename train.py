import os
import sys
sys.path.append('../metrics/')
import argparse
import pickle
import glob
import numpy as np
import torch
import torch.nn.functional as F
import time
import torch.nn as nn
import logging
import random
import dgl
from net import STNet
from utils import *
from config import *
from metrics import *


parser = argparse.ArgumentParser(description='model')
parser.add_argument('--gpu', type=str, default='6', help='which gpu to use.')
parser.add_argument('--state', type=str, default='def', help='exp statement.')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')
parser.add_argument('--encuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--density', action='store_true', default=False, help='calculate density.')
parser.add_argument('--load', action='store_true', default=False, help='load parms.')
parser.add_argument('--cityname', type=str, default='zhuzhou', help='dataset of city.')
parser.add_argument('--model', type=str, default='AsyncSTGCN', help='name of baseline model.')
parser.add_argument('--loss', type=str, default='l1', help='loss.')
parser.add_argument('--te', type=str, default='combine', choices=['combine', 'ind', 'share'], help='name of baseline model.')
parser.add_argument('--decoder', type=str, default='GRUMLP', choices=['GRUMLP', 'MLP'], help='decoder')
parser.add_argument('--seed', type=int, default=6, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of batch to train and test.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hid_dim', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--te_dim', type=int, default=16, help='Number of hidden units for time encoding.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--beta', type=float, default=1, help='coefficient of period')
parser.add_argument('--n_output', type=int, default=12, help='prediction length per decoding step')
parser.add_argument('--patience', type=int, default=10, help='Patience')
args = parser.parse_args()
args.pid = os.getpid()
# torch.set_num_threads(2)

loader_train, loader_val, loader_test  = load_data(args)
print("# of lanes", args.N)

if(args.debug):
    args.state = "debug"
    logging.basicConfig(level = logging.INFO,filename='../log/{}_debug'.format(args.model),\
                    filemode='{}'.format(args.logmode),\
                    format = '%(message)s')
else:
    logging.basicConfig(level = logging.INFO,filename='../log/{}_{}_{}'.\
                        format(args.cityname, args.model, args.state),\
                        filemode='{}'.format(args.logmode),\
                        format = '%(message)s')
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.encuda and torch.cuda.is_available():
    print("cuda is available.")
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    print("cuda is NOT available!")
    args.device = torch.device('cpu')

print(args)
logger.info(args)


def train_epoch(data_loader):
    """
    :input: (n_samples, 2(X or Y), n_nodes(N), L_steps(alterable), n_features)
    """
    net.train()
    for i, batch_data in enumerate(data_loader):
        optimizer.zero_grad()
        X, Y, mask_X, mask_Y, X_edges, X_edges_feats, Y_edges, Y_edges_feats = batch_data
        X_edges, X_edges_feats, Y_edges, Y_edges_feats = X_edges[0], X_edges_feats[0], Y_edges[0], Y_edges_feats[0]
        B, N, Lx, _ = X.shape
        B, N, Ly, _ = Y.shape
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        mask_X = mask_X.to(args.device, non_blocking=True)
        mask_Y = mask_Y.to(args.device, non_blocking=True)
        ### model prediction ###
        # adj_X = dgl.graph((X_edges[0],X_edges[1]), num_nodes=N*Lx, device=args.device)
        adj_X = dgl.graph((X_edges[0],X_edges[1]), num_nodes=N*Lx).to(args.device, non_blocking=True)
        adj_X.edata['feature']=X_edges_feats.to(args.device, non_blocking=True)
        # adj_Y = dgl.graph((Y_edges[0],Y_edges[1]), num_nodes=N*Lx+N, device=args.device)
        adj_Y = dgl.graph((Y_edges[0],Y_edges[1]), num_nodes=N*Lx+N).to(args.device, non_blocking=True)
        adj_Y.edata['feature']=Y_edges_feats.to(args.device, non_blocking=True)
        adjs = (adj_X, adj_Y)
        ### model prediction ###
        pred_period, pred_flow, _ = net(X, mask_X, Y, adjs) # (N, Ly)
        pred_period = pred_period[:,:Ly]
        pred_flow = pred_flow[:,:Ly]
        # print(pred_period.shape, pred_flow.shape)
        y_period, y_flow, y_stime = Y[...,0].squeeze(dim=0), Y[...,1].squeeze(dim=0), Y[...,2].squeeze(dim=0)
        loss_time, loss_period, loss_stime = mask_period_metric(pred_period, y_period, y_stime, mask_Y.squeeze(dim=0), reduction='mean', metric='l1')
        loss_flow = mask_flow_metric(pred_flow*y_period, y_flow*y_period, mask_Y.squeeze(dim=0), reduction='mean', metric='l1')
        loss = loss_time + loss_flow*args.beta
        loss.backward()
        optimizer.step()

    return loss.item(), loss_time.item(), loss_period.item(), loss_stime.item(), loss_flow.item()

def test_epoch(data_loader, is_test=False):
    net.eval()
    loss_time = loss_flow = 0
    l1_time = l1_period = l1_stime = l1_flow = .0
    l2_time = l2_period = l2_stime = l2_flow = .0
    n_value = 0
    l1_flow_density = n_second = 0
    for i, batch_data in enumerate(data_loader):
        X, Y, mask_X, mask_Y, X_edges, X_edges_feats, Y_edges, Y_edges_feats = batch_data
        X_edges, X_edges_feats, Y_edges, Y_edges_feats = X_edges[0], X_edges_feats[0], Y_edges[0], Y_edges_feats[0]
        B, N, Lx, _ = X.shape
        B, N, Ly, _ = Y.shape
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        mask_X = mask_X.to(args.device, non_blocking=True)
        mask_Y = mask_Y.to(args.device, non_blocking=True)
        # adj_X = dgl.graph((X_edges[0],X_edges[1]), num_nodes=N*Lx, device=args.device)
        adj_X = dgl.graph((X_edges[0],X_edges[1]), num_nodes=N*Lx).to(args.device, non_blocking=True)
        adj_X.edata['feature']=X_edges_feats.to(args.device, non_blocking=True)
        # adj_Y = dgl.graph((Y_edges[0],Y_edges[1]), num_nodes=N*Lx+N, device=args.device)
        adj_Y = dgl.graph((Y_edges[0],Y_edges[1]), num_nodes=N*Lx+N).to(args.device, non_blocking=True)
        adj_Y.edata['feature']=Y_edges_feats.to(args.device, non_blocking=True)
        adjs = (adj_X, adj_Y)
        pred_period_test, pred_flow, pred_flow_test = net(X, mask_X, Y, adjs, is_test) # (N, Ly)
        pred_period_test = torch.clip(pred_period_test, 0.05, 1)
        pred_period = pred_period_test[:,:Ly]
        pred_flow = pred_flow[:,:Ly]
        y_period, y_flow, y_stime = Y[...,0].squeeze(dim=0), Y[...,1].squeeze(dim=0), Y[...,2].squeeze(dim=0)
        loss_time_batch, _, _, _ = mask_period_metric(pred_period, y_period, y_stime, mask_Y.squeeze(dim=0), reduction='sum', metric='l1')
        loss_flow_batch, _ = mask_flow_metric(pred_flow*y_period, y_flow*y_period, mask_Y.squeeze(dim=0), reduction='sum', metric='l1')
        loss_time += loss_time_batch
        loss_flow += loss_flow_batch

        pred_period_recover = pred_period*300
        y_period_recover = y_period*300
        y_stime_recover = y_stime*300
        l1_time_b, l1_period_b, l1_stime_b, _ = mask_period_metric(pred_period_recover, y_period_recover, y_stime_recover, mask_Y.squeeze(dim=0), reduction='sum', metric='l1')
        l1_time += l1_time_b
        l1_period += l1_period_b
        l1_stime += l1_stime_b
        l2_time_b, l2_period_b, l2_stime_b, _ = mask_period_metric(pred_period_recover, y_period_recover, y_stime_recover, mask_Y.squeeze(dim=0), reduction='sum', metric='l2')
        l2_time += l2_time_b
        l2_period += l2_period_b
        l2_stime += l2_stime_b

        pred_flow_recover = (pred_flow * std_unitflow + mean_unitflow) * y_period_recover # per minute
        pred_flow_recover = torch.clip(pred_flow_recover, 0, 1e8)
        y_flow_recover = (y_flow * std_unitflow + mean_unitflow) * y_period_recover 
        l1_sum, _ = mask_flow_metric(pred_flow_recover, y_flow_recover, mask_Y.squeeze(dim=0), reduction='sum', metric='l1')
        l1_flow += l1_sum
        l2_sum, n_value_batch = mask_flow_metric(pred_flow_recover, y_flow_recover, mask_Y.squeeze(dim=0), reduction='sum', metric='l2')
        l2_flow += l2_sum
        n_value += n_value_batch

        if(is_test):
            ### flow density error
            pred_flow_density = pred_flow_test * std_unitflow + mean_unitflow
            pred_flow_density = torch.clip(pred_flow_density, 0, 1e8)
            y_flow_density = y_flow * std_unitflow + mean_unitflow
            l1_flow_density_sum, n_second_batch = mask_flowdensity_metric(pred_flow_density, pred_period_test*300, y_flow_density, y_period_recover, mask_Y.squeeze(dim=0))
            l1_flow_density += l1_flow_density_sum
            n_second += n_second_batch

    # print(l1,l2,n_value)
    l1_time = (l1_time/n_value).item()
    l1_period = (l1_period/n_value).item()
    l1_stime = (l1_stime/n_value).item()
    l1_flow = (l1_flow/n_value).item()
    l2_time = np.sqrt((l2_time/n_value).item())
    l2_period = np.sqrt((l2_period/n_value).item())
    l2_stime = np.sqrt((l2_stime/n_value).item())
    l2_flow = np.sqrt((l2_flow/n_value).item())
    loss_time = (loss_time/n_value).item()
    loss_flow = (loss_flow/n_value).item()
    if(is_test):
        l1_flow_density = (l1_flow_density/n_second).item()*60
        # print("L1_flow_density_minute:{:.4f}".format(l1_flow_density))
    return (loss_time, loss_flow*args.beta), (l1_time, l1_period, l1_stime, l1_flow, l1_flow_density), (l2_time, l2_period, l2_stime, l2_flow)

def print_log(mae, rmse, loss, state):
    state_str = "{} -------------------------------------".format(state)
    loss_str = "Loss (o,t,f): {:.4f}, {:.4f}, {:.4f}".format(loss[0]+loss[1], loss[0], loss[1])
    mae_str = "MAE (t,p,st,f,fd): {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(mae[0], mae[1], mae[2], mae[3], mae[4])
    rmse_str = "RMSE (t,p,st,f): {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(rmse[0], rmse[1], rmse[2], rmse[3])
    print(state_str)
    print(loss_str)
    print(mae_str)
    print(rmse_str)
    logger.info(state_str)
    logger.info(loss_str)
    logger.info(mae_str)
    logger.info(rmse_str)

print("Model:", args.model)
net = STNet(args).to(device=args.device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

min_val_loss = 1e15
best_epoch = 0

'''load model'''
if(args.load):
    PARAM_SAVE_PATH = './model_paras_{}/{}/'.format(args.cityname, args.model)
    files = glob.glob('{}{}_{}_{}_{}-epoch_*.pkl'.format(PARAM_SAVE_PATH, args.model, args.state, args.n_output, args.hid_dim))
    files.sort()
    print(files)
    param_file = files[-1]
    print("load params:", param_file)
    paras_load = torch.load(param_file)
    net.load_state_dict(paras_load['model'])
    best_epoch = paras_load['epoch']
    args.state = args.state + "load"

for epoch in range(best_epoch, args.epochs):
    print("Epoch: {} ====================================================".format(epoch+1))
    logger.info("Epoch: {} ====================================================".format(epoch+1))
    st_time = time.time()
    print('training......')
    train_loss = train_epoch(loader_train)
    print("Train loss (o,t,p,st,f): {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4]))
    logger.info("Train loss (o,t,p,st,f): {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4]))
    
    with torch.no_grad():
        print('validating......')
        val_loss, val_mae, val_rmse = test_epoch(loader_val)
    
    # with torch.no_grad():
    #     print('testing......')
    #     test_loss, test_mae, test_rmse = test_epoch(loader_test, is_test=args.density)

    comb_val_loss = val_loss[0] + val_loss[1]
    if(comb_val_loss < min_val_loss):
        min_val_loss = comb_val_loss
        best_epoch = epoch + 1
        
        with torch.no_grad():
            print('testing......')
            test_loss, test_mae, test_rmse = test_epoch(loader_test, is_test=args.density)
            
        best_loss = test_loss
        best_mae = test_mae
        best_rmse = test_rmse

        '''save model'''
        PARAM_SAVE_PATH = './model_paras_{}/{}/'.format(args.cityname, args.model)
        if not os.path.exists(PARAM_SAVE_PATH):
            os.makedirs(PARAM_SAVE_PATH)
        paras_save = {
            'model':net.state_dict(),
            'epoch':best_epoch,
        }
        files = glob.glob('{}{}_{}_{}_{}-epoch_*.pkl'.format(PARAM_SAVE_PATH, args.model, args.state, args.n_output, args.hid_dim))
        files.sort()
        for file in files[:-4]: # preserve latest 3 
            os.remove(file)
        save_path = '{}{}_{}_{}_{}-epoch_{:03d}.pkl'.format(PARAM_SAVE_PATH, args.model, args.state, args.n_output, args.hid_dim, best_epoch)
        torch.save(paras_save, save_path)
        
    print_log(val_mae,val_rmse,val_loss,'Val')
    # print_log(test_mae,test_rmse,test_loss,'Test')
    print_log(best_mae,best_rmse,best_loss,'Best Epoch-{}'.format(best_epoch))
    print('time: {:.4f}s\n'.format(time.time() - st_time))
    logger.info('time: {:.4f}s\n\n'.format(time.time() - st_time))

    # model stop condition
    if(epoch+1 - best_epoch >= args.patience):
        sys.exit(0)
        
