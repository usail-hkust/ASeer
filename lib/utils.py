# coding:utf-8
import numpy as np
import torch
import pandas as pd
import codecs
import sys
import pickle
import time
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import *

class GetDataset(Dataset):
    def __init__(self, X):

        self.X = X   # numpy.ndarray (n_sample, n_nodes(N), L_steps(alterable), n_features)
        
    def __getitem__(self, index):
        
        # torch.Tensor
        tensor_X = self.X[index]
        
        return tensor_X

    def __len__(self):
        return len(self.X)
    
def make_dataset(raw_data, distance, reachability):
    """
    :input: raw_data (n_samples, (X, Y, mask_X, mask_Y, ...), n_nodes(N), L_steps(alterable), n_features)
    :distance: distance between sensors
    :reachability: lane-level road-network reachability between sensors
    :return: dataset
    x features: ed_time, period, flow, unit_flow
    y features: st_time, period, flow, unit_flow
    """
    mean_unitflow, std_unitflow = norm_param[cityname]
    data_set = []
    for x, y, mask_x, mask_y, x_edges, y_edges in raw_data:
        # print(x[...,-1].mean(), y[...,-1].mean())
        time_baseline = (x[...,0].max()//3600+1)*3600
        time_baseline_x = time_baseline - 3600

        x = np.array(x, dtype=np.float64)
        x_view = x.reshape(-1, x.shape[-1])
        x_period = x[...,1]/300 # period
        x_unit_flow = (x[...,-1]-mean_unitflow)/std_unitflow # unit_flow
        x_delta_t = (x[:,-1:,0]-x[...,0])/300 * mask_x
        # print("x_delta_t:", x_delta_t.max(), x_delta_t.min())
        x_feat = np.stack([x_period, x_unit_flow, x_delta_t], axis=-1)

        y = np.array(y, dtype=np.float64)
        y_period = y[...,1]/300 # period
        y_unit_flow = (y[...,-1]-mean_unitflow)/std_unitflow # unit_flow
        etime_x_last = x[:,-1:,0] # (N,1)
        etime_x_last = np.where(etime_x_last<1e-8, time_baseline_x, etime_x_last)
        y_delta_t = np.clip((y[...,0]-etime_x_last), 0, 1e8) / 300
        # print(list((y[:,1:,0] - y[:,:-1,1])[5]))
        # print(y[...,0].astype("int"))
        # print(y_delta_t.shape, (y_delta_t[:5,:]*300).astype("int"))
        # print("y_delta_t:", y[...,0].max(), y_delta_t.max(), y_delta_t.min())
        y_feat = np.stack([y_period, y_unit_flow, y_delta_t], axis=-1)
        
        Nx, Lx, Fx = x.shape
        x_src_lane = x_edges[0,:]//Lx
        x_dst_lane = x_edges[1,:]//Lx
        x_distance = distance[x_src_lane, x_dst_lane]/1000
        x_reachability = reachability[x_src_lane, x_dst_lane]
        x_edge_dt = (x_view[x_edges[1,:]][:,0] - x_view[x_edges[0,:]][:,0])/300
        # print("x_edge_dt:", x_edge_dt.mean(), x_edge_dt.max(), x_edge_dt.min())
        x_edges_feats = np.stack([x_distance, x_edge_dt, x_reachability, x_dst_lane], axis=-1).astype(np.float32)
        # print(x_edges_feats.max(axis=0))
        
        Ny, Ly, Fy = y.shape
        y_src_lane = y_edges[0,:]//Lx
        y_dst_lane = y_edges[1,:]-Nx*Lx
        y_distance = distance[y_src_lane, y_dst_lane]/1000
        y_reachability = reachability[y_src_lane, y_dst_lane]
        y_edge_dt = (x_view[y_edges[0,:]][:,0] - etime_x_last[y_dst_lane][:,0])/300
        # print(y_edge_dt.tolist())
        # print("y_edge_dt:", y_edge_dt.mean(), y_edge_dt.max(), y_edge_dt.min())
        # print(x_edge_dt.shape, y_edge_dt.shape)
        y_edges_feats = np.stack([y_distance, y_edge_dt, y_reachability, y_dst_lane], axis=-1).astype(np.float32)
        # print(x.dtype, y.dtype, mask_x.dtype, mask_y.dtype, x_edges.dtype, x_edges_feats.dtype, y_edges.dtype, y_edges_feats.dtype)
        # print(x_edges.shape, x_edges_feats.shape)
        
        x_tensor = torch.from_numpy(x_feat).float()
        y_tensor = torch.from_numpy(y_feat).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        mask_y_tensor =  torch.from_numpy(mask_y)
        x_edges_tensor = torch.from_numpy(x_edges)
        x_edges_feats_tensor = torch.from_numpy(x_edges_feats)
        y_edges_tensor = torch.from_numpy(y_edges)
        y_edges_feats_tensor = torch.from_numpy(y_edges_feats)
        data_set.append([x_tensor,y_tensor,mask_x_tensor,mask_y_tensor,x_edges_tensor,x_edges_feats_tensor,y_edges_tensor,y_edges_feats_tensor])
    
    return GetDataset(data_set)


def load_data(args):
    print("PID:", args.pid)
    st_time = time.time()
    global cityname 
    cityname = args.cityname
    
    # load data
    with open("../data/{}/dataset_onhour.pkl".format(args.cityname),"rb") as fp:
        data_xy = pickle.load(fp)
    with open("../data/{}/message_edges.pkl".format(args.cityname),"rb") as fp:
        message_edges = pickle.load(fp)
    distance = np.load("../data/{}/distance_geo.npy".format(cityname))
    reachability = np.load("../data/{}/reachability.npy".format(cityname))

    data_hour = []
    for xy, edges in zip(data_xy, message_edges):
        data_hour.append(xy + edges)
    L = len(data_hour)
    print("Length of dataset:", L)
    train_data = data_hour[:int(L*0.6)] # 622
    val_data = data_hour[int(L*0.6):int(L*0.8)]
    test_data = data_hour[int(L*0.8):]
    args.N = train_data[0][0].shape[0]
    print('Data load time: {:.4f}s'.format(time.time() - st_time))

    dataset_train = make_dataset(train_data, distance, reachability)
    dataset_val = make_dataset(val_data, distance, reachability)
    dataset_test = make_dataset(test_data, distance, reachability)
    # loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
    # loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)
    # loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)
    n_workers = 1
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    
    print("# of dataset split:", len(loader_train), len(loader_val), len(loader_test))
    print("==================== load_data finished. ====================")
    del dataset_train
    del dataset_val
    del dataset_test
    return loader_train, loader_val, loader_test 


