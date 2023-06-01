import os
import numpy as np
import torch

def mask_period_metric(period_pred, period_y, stime_y, mask, reduction='mean', metric='l1'):
    n_value = mask.sum()
    # print(period_pred.shape, period_y.shape, stime_y.shape, mask.shape)
    N, ly = period_y.shape
    # period_pred = pred[:,:ly]
    tril_mask = torch.tril(torch.ones(ly,ly).cuda()).unsqueeze(dim=0)
    pred_expand = period_pred.unsqueeze(dim=-2).repeat(1,ly,1)
    stime_init = stime_y[:,:1]
    # print(pred_expand.shape, tril_mask.shape)
    stime_pred = torch.sum(pred_expand*tril_mask, dim=-1)[:,:ly-1] + stime_init # (N, ly-1)
    stime_pred = torch.cat([stime_init, stime_pred], dim=-1) # (N, ly)

    ### sum of ground truth periods
    # stime = torch.sum(period_y.unsqueeze(dim=-2).repeat(1,ly,1)*tril_mask, dim=-1)[:,:ly-1] + stime_init # (N, ly-1)
    # stime = torch.cat([stime_init, stime], dim=-1)
    # if(reduction == 'sum'):
    #     res = torch.abs(stime - stime_y)*mask
    #     sum_error = torch.sum(res)
    #     if(sum_error>1):
    #         print(sum_error, sum_error/n_value)
            # print(res)

    if(reduction == 'mean'):
        if(metric == "l1"):
            loss_period = torch.sum(torch.abs(period_pred - period_y)*mask)/n_value
            loss_stime = torch.sum(torch.abs(stime_pred - stime_y)*mask)/n_value 
            loss = loss_period + loss_stime
        elif(metric == "l2"):
            loss_period = torch.sum(torch.pow(period_pred - period_y, 2)*mask)/n_value
            loss_stime = torch.sum(torch.pow(stime_pred - stime_y, 2)*mask)/n_value 
            loss = loss_period + loss_stime
        else:
            print("Invalid metric!")
            exit(0)
        return loss, loss_period, loss_stime
    elif(reduction == 'sum'):
        if(metric == "l1"):
            loss_period = torch.sum(torch.abs(period_pred - period_y)*mask)
            loss_stime = torch.sum(torch.abs(stime_pred - stime_y)*mask)
            loss = loss_period + loss_stime
            # print("l1:", loss.item(), loss_period.item(), loss_stime.item(), n_value.item())
        elif(metric == "l2"):
            loss_period = torch.sum(torch.pow(period_pred - period_y, 2)*mask)
            loss_stime = torch.sum(torch.pow(stime_pred - stime_y, 2)*mask)
            loss = loss_period + loss_stime
        elif(metric == "ape"):
            loss_period = torch.sum(torch.abs(period_pred - period_y)/(period_y+1e-6)*mask)
            loss_stime = torch.sum(torch.abs(stime_pred - stime_y)/(stime_y+1e-6)*mask)
            loss = loss_period + loss_stime
            # print("period_pred:", period_pred*mask)
            # print("period_y:", period_y*mask)
            # print("period_errors:", torch.abs(period_pred - period_y)/(period_y+1e-6)*mask)
            # print("ape:", loss.item(), loss_period.item(), loss_stime.item(), n_value.item())
            # print("max:",((1/(period_y+1e-6))*mask).max(),((1/(stime_y+1e-6))*mask).max())
        else:
            print("Invalid metric!")
            exit(0)
        return loss, loss_period, loss_stime, n_value
    else:
        print("Invalid reduction!")


def mask_flow_metric(pred, truth, mask, reduction='mean', metric="l1"):
    n_value = mask.sum()
    # print(n_value, pred.shape, truth.shape, mask.shape)
    N, ly = truth.shape
    # pred = pred[:,:ly]
    if(reduction == 'mean'):
        if(metric == "l1"):
            loss = torch.sum(torch.abs(pred - truth)*mask)/n_value
        elif(metric == "l2"):
            loss = torch.sum(torch.pow(pred - truth, 2)*mask)/n_value
        else:
            print("Invalid metric!")
            exit(0)
        return loss

    elif(reduction == 'sum'):
        if(metric == "l1"):
            loss = torch.sum(torch.abs(pred - truth)*mask)
        elif(metric == "l2"):
            loss = torch.sum(torch.pow(pred - truth, 2)*mask)
        elif(metric == "ape"):
            mask_zero =  (torch.abs(truth) > 1e-3)
            mask_new = mask * mask_zero
            loss = torch.sum(torch.abs(pred - truth)/(truth+1e-6)*mask_new)
            n_value = mask_new.sum()
            # print("max:",((torch.abs(pred - truth)/(truth+1e-6))*mask_new).max())
            # print("max2:",((torch.abs(pred - truth)/(truth+1e-6))*mask).max())
        else:
            print("Invalid metric!")
            exit(0)
        return loss, n_value
    else:
        print("Invalid reduction!")

        
def mask_flowdensity_metric(pred_flow_density, pred_period, y_flow_density, y_period, mask):
    # print(n_value, pred.shape, truth.shape, mask.shape)
    N, ly = y_period.shape
    # pred = pred[:,:ly]
    
    # y_flow_second = torch.zeros(N, 7200).to(args.device)
    # tril_mask = torch.tril(torch.ones(ly,ly)).unsqueeze(dim=0).to(args.device) # (1, ly, ly)
    # period_expand = y_period.unsqueeze(dim=-2).repeat(1,ly,1)
    # rows = torch.arange(N).unsqueeze(dim=-1).repeat(1,ly).flatten()
    # cols = torch.sum(period_expand*tril_mask, dim=-1).flatten().to(torch.int32) # (N*ly,)
    # y_flow_second[rows, cols] = y_flow_density

    # y_flow_density = y_flow_density.numpy()
    # y_period = y_period.numpy()
    # mask = mask.numpy()
    y_period = y_period.long()
    # print(list(y_period[0]))
    y_flow_second = torch.zeros((N, 7200)).cuda()
    mask_second = torch.zeros((N, 7200), dtype=torch.bool).cuda()
    for i in range(N):
        y_flow_per = y_flow_density[i]
        y_period_per = y_period[i]
        mask_per = mask[i]
        y_flow_second_per = y_flow_per.repeat_interleave(y_period_per)
        y_flow_second[i, :len(y_flow_second_per)] = y_flow_second_per
        mask_second[i, :len(y_flow_second_per)] = mask_per.repeat_interleave(y_period_per)

    # pred_flow_density = pred_flow_density.numpy()
    # pred_period = np.around(pred_period.numpy())
    pred_period = torch.round(pred_period).long()
    pred_flow_second = torch.zeros((N, 7200)).cuda()
    for i in range(N):
        pred_flow_per = pred_flow_density[i]
        pred_period_per = pred_period[i]
        pred_flow_second_per = pred_flow_per.repeat_interleave(pred_period_per)[:7200]
        pred_flow_second[i, :len(pred_flow_second_per)] = pred_flow_second_per
    error = torch.sum(torch.abs(pred_flow_second - y_flow_second)*mask_second)
    n_second = mask_second.sum()
    return error, n_second
    