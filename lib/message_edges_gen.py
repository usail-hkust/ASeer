import pickle 
import numpy as np
import sys

args = sys.argv
cityname = args[1]
print("Dataset:", cityname)

with open(f"../data/{cityname}/dataset_onhour.pkl","rb") as fp:
    data_hour_0 = pickle.load(fp)
distance = np.load(f"../data/{cityname}/distance_geo.npy")
reachability = np.load(f"../data/{cityname}/reachability.npy")

for i in range(distance.shape[0]):
    distance[i,i] = 1e6
    reachability[i,i] = 0

N = distance.shape[0]
dist_eps = 1000
neibors = [[] for i in range(N)]
n_neibors = []
for i in range(N):
    for j in range(N):
        if(distance[i,j]<=dist_eps):
            neibors[i].append(j)
    n_neibors.append(len(neibors[i]))
# neibors

max_dt = -1e8
# N = data_hour_0[0][0].shape[0]
for i in range(len(data_hour_0)):
    for n in range(N):
        decoder_st = data_hour_0[i][1][n][0][0]
        encoder_et = data_hour_0[i][0][n][-1][0]
        dt = decoder_st - encoder_et
        max_dt = max(max_dt, dt)
        if(dt < 0 and decoder_st!=0):
            print(encoder_et, decoder_st, dt)
print(max_dt)

### store all msg

# features: st_time, ed_time, mid_time, period, flow, unit_flow
# x new features: ed_time, period, flow, unit_flow
# y new features: st_time, period, flow, unit_flow

print("len of dataset:", len(data_hour_0))
lane_id = np.arange(data_hour_0[0][0].shape[0])
lane_id = lane_id[:,np.newaxis,np.newaxis]
data_hour_0_graph = []
message_edges = []
for batch_i, batch_data in enumerate(data_hour_0[:]):
    if(batch_i%10 == 0):
        print(batch_i)
    X_ori, Y_ori, mask_X_ori, mask_Y_ori = batch_data
    # X_ori[...,:3] = X_ori[...,:3].astype('int32')
    # X_ori[...,3:5] = X_ori[...,3:5].astype('int16')
    # X_ori[...,-1] = X_ori[...,-1].astype('int32')
    # Y_ori[...,:3] = Y_ori[...,:3].astype('int32')
    # Y_ori[...,3:5] = Y_ori[...,3:5].astype('int16')
    # Y_ori[...,-1] = Y_ori[...,-1].astype('int32')
    x_et = X_ori[...,0]
    x_p = X_ori[...,1]
    x_flow = X_ori[...,2]
    x_unitflow = X_ori[...,3]
    x_st = np.where(x_et > 0, x_et - x_p + 1, 0)
    x_mt = np.where(x_et > 0, (x_st+x_et)/2, 0)
    X_ori = np.stack([x_st, x_et, x_mt, x_p, x_flow, x_unitflow], axis=-1)

    y_st = Y_ori[...,0]
    y_p = Y_ori[...,1]
    y_flow = Y_ori[...,2]
    y_unitflow = Y_ori[...,3]
    y_et = np.where(y_st > 0, y_st + y_p - 1, 0)
    y_mt = np.where(y_st > 0, (y_st+y_et)/2, 0)
    Y_ori = np.stack([y_st, y_et, y_mt, y_p, y_flow, y_unitflow], axis=-1)

    ### edges of X 
    # print(X_ori.shape, lane_id.repeat(X_ori.shape[1],axis=1).shape)
    X = np.concatenate([X_ori,lane_id.repeat(X_ori.shape[1],axis=1)],axis=-1)
    X = X.reshape(-1,X.shape[-1])
    mask_X = mask_X_ori.reshape(-1)

    argsort_time = np.argsort(X[:,1]) # *** sort by endtime
    X = X[argsort_time] # sorted X by endtime
    mask_X = mask_X[argsort_time]
    
    ### msg propogation
    msg_list = [[] for i in range(X.shape[0])]
    for i, flow in enumerate(X):
        if(mask_X[i] == 0): 
            continue
        ori_index = argsort_time[i]
        node_id = flow[-1]
        end_time = flow[1]
#         i_msgs = msg_list[node_id]
        # msg propagation
        neibor_nodes = neibors[node_id]
        msg_index = ori_index
        for neibor_id in neibor_nodes:
            msg_list[neibor_id].append([msg_index, end_time, flow])

    X_edges = [[],[]]
    X_edges_feats = []

#         L = 0
#         for i in range(N):
#             L += len(msg_list[i])
#         print("before X:", L/N)
    for i, flow in enumerate(X):
        if(mask_X[i] == 0): 
            continue
        ori_index = argsort_time[i]
        node_id = flow[-1]
        end_time = flow[1]
        i_msgs = msg_list[node_id]
        if(len(i_msgs)>0):
            for n, msg in enumerate(i_msgs):
                msg_index, msg_time, msg_value = msg
                delta_t = end_time - msg_time
                if(delta_t<0):
                    break
                dist = distance[msg_value[-1],node_id]
                X_edges[0].append(msg_index)
                X_edges[1].append(ori_index)
                X_edges_feats.append([dist,delta_t])
            msg_list[node_id] = i_msgs[n:]
    X_edges = np.array(X_edges)
    X_edges_feats = np.array(X_edges_feats)

    ### edges of Y
#         L = 0
#         for i in range(N):
#             L += len(msg_list[i])
#         print("After X:", L/N)

    Y = np.concatenate([Y_ori,lane_id.repeat(Y_ori.shape[1],axis=1)],axis=-1)
    Y = Y.reshape(-1,Y.shape[-1])
    mask_Y = mask_Y_ori.reshape(-1)
    Y_edges = [[],[]]
    Y_edges_feats = []
#     for i, flow in enumerate(Y):
#         if(mask_Y[i] == 0): 
#             continue
#         ori_index = i + len(X)
#         node_id = flow[-1]
#         end_time = flow[1]
#         i_msgs = msg_list[node_id]
#         if(len(i_msgs)>0):
#             for msg in i_msgs:
#                 msg_index, msg_time, msg_value = msg
#                 delta_t = end_time - msg_time
#                 dist = distance[msg_value[-1],node_id]
#                 Y_edges[0].append(msg_index)
#                 Y_edges[1].append(ori_index)
#                 Y_edges_feats.append([dist,delta_t])

    for i in range(N):
        ori_index = i + len(X)
        node_id = i
        i_msgs = msg_list[node_id]
        if(len(i_msgs)>0):
            for msg in i_msgs:
                msg_index, msg_time, msg_value = msg
                end_time = ((msg_time//3600)+1)*3600
                delta_t = end_time - msg_time
                dist = distance[msg_value[-1],node_id]
                Y_edges[0].append(msg_index)
                Y_edges[1].append(ori_index)
                Y_edges_feats.append([dist,delta_t])
    Y_edges = np.array(Y_edges)
    Y_edges_feats = np.array(Y_edges_feats)
#         data_hour_0_graph.append([X_ori, Y_ori, mask_X_ori, mask_Y_ori, X_edges, X_edges_feats, Y_edges, Y_edges_feats])
    data_hour_0_graph.append([X_ori, Y_ori, mask_X_ori, mask_Y_ori, X_edges.astype(np.int32), Y_edges.astype(np.int32)])
    message_edges.append([X_edges.astype(np.int32), Y_edges.astype(np.int32)])


with open(f"../data/{cityname}/message_edges.pkl","wb") as fp:
    pickle.dump(message_edges, fp)

