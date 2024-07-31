import os

import numpy as np

from RNN_iSLR import RNNiSLR
from load_data import iter_lms_KBD, iter_lms_TIS
from data_util_fn import MTRiSLRDataset, train_test_split, fit_iterative_SLR
import torch as tc
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import time


def train_one_epoch(dl:tc.utils.data.DataLoader, model:RNNiSLR, loss:tc.nn.modules.loss, optim:tc.optim):
    loss_in_epoch = 0
    model.train(True)

    if dl.dataset.data_incycle is not None:
        _, xb_sub, _ = next(iter(dl))
        Nsubbatches = int(xb_sub.size(-1) / 3) + 2
        num_subintervals = int(xb_sub.size(-1) / 3) + 1
    else:
        Nsubbatches, num_subintervals = 0,0

    for idx, xb in enumerate(dl):
        # dl_iter.set_description("Training on epoch...")
        if dl.dataset.data_incycle is None and dl.dataset.peak_features is None:
            xb = xb.swapdims(1,0).to(model.device)
            optim.zero_grad()
            lb = loss(model.forward(xb[:-1, :, :]), xb[-1, :, :])
            lb.backward()
            optim.step()
        elif dl.dataset.data_incycle is not None:
            # xbb: [batch_size, seq_len, 3], xb_sub: [batch_size, 3*num_subintervals], xb_peak: [batch_size, seq_len, num_peak_features]
            xbb, xb_sub, xb_peak = xb
            xbb, xb_sub, xb_peak = xbb.to(model.device), xb_sub.to(model.device), xb_peak.to(model.device)
            xb_tgt = xbb[:,-1,:] # xb_tgt: [batch_size, 3]
            # Nsubbatches = int(xb_sub.size(-1)/3) + 2
            # num_subintervals = int(xb_sub.size(-1)/3) + 1
            lb=tc.zeros(1).to(model.device)
            for sb in range(Nsubbatches):
                if sb == 0:
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=",xbb[:,:-1,:].shape)
                    input_b = tc.concatenate([xbb[:, :-1, :], tc.zeros([xbb.size(0), 1, 3], device=model.device)], dim=1)
                elif (sb > 0) and (sb < (Nsubbatches-1)):
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=", xbb[:, :-1, :].shape)
                    # print("xb_sub slice shape=", xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3).shape)
                    input_b = tc.concatenate([xbb[:, :-1, :],
                                              xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3)], dim=1)
                else:
                    input_b = xbb # input_b: [batch_size, seq_len, 3]

                # if dl.dataset.peak_features is not None:
                #     input_b = tc.concatenate([input_b, xb_peak], dim=-1) # input_b: [batch_size, seq_len, 3+num_peak_features]

                optim.zero_grad()
                lb_sub = loss(model.forward(x=input_b.swapdims(1,0), y=xb_peak.swapdims(1,0)), xb_tgt)
                lb_sub.backward()
                optim.step()
                lb += lb_sub
            lb = lb/Nsubbatches
        else:
            print("dl.dataset.data_incycle cannot be None when dl.dataset.peak.features is not None!")
            exit()
        loss_in_epoch += lb.detach().item()
    return loss_in_epoch/(idx+1)


def eval_model(dl:tc.utils.data.DataLoader, model:RNNiSLR, loss:tc.nn.modules.loss):
    eval_loss = 0
    model.train(False)
    pred = tc.tensor([], device=model.device)

    if dl.dataset.data_incycle is not None:
        _, xb_sub,_ = next(iter(dl))
        Nsubbatches = int(xb_sub.size(-1) / 3) + 2
        num_subintervals = int(xb_sub.size(-1) / 3) + 1
    else:
        Nsubbatches, num_subintervals = 0,0

    for idx, xb in enumerate(dl):
        # dl_iter.set_description("Evaluating on epoch...")
        if dl.dataset.data_incycle is None and dl.dataset.peak_features is None:
            xb = xb.swapdims(1,0).to(model.device)
            x_pred = model.forward(xb[:-1, :, :])
            pred = tc.concatenate([pred, x_pred.detach()])
            lb = loss(x_pred, xb[-1, :, :])

        elif dl.dataset.data_incycle is not None:
            # xbb: [batch_size, seq_len, 3], xb_sub: [batch_size, 3*num_subintervals], xb_peak: [batch_size, seq_len, num_peak_features]
            xbb, xb_sub, xb_peak = xb
            xbb, xb_sub, xb_peak = xbb.to(model.device), xb_sub.to(model.device), xb_peak.to(model.device)
            xb_tgt = xbb[:,-1,:] # xb_tgt: [batch_size, 3]
            # Nsubbatches = int(xb_sub.size(-1)/3) + 2
            # num_subintervals = int(xb_sub.size(-1)/3) + 1
            lb=tc.zeros(1).to(model.device)
            xb_preds = tc.zeros([Nsubbatches, xb_tgt.size(0), 3])
            for sb in range(Nsubbatches):
                if sb == 0:
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=",xbb[:,:-1,:].shape)
                    input_b = tc.concatenate([xbb[:, :-1, :], tc.zeros([xbb.size(0), 1, 3], device=model.device)], dim=1)
                elif (sb > 0) and (sb < (Nsubbatches-1)):
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=", xbb[:, :-1, :].shape)
                    # print("xb_sub slice shape=", xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3).shape)
                    input_b = tc.concatenate([xbb[:, :-1, :],
                                              xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3)], dim=1)
                else:
                    input_b = xbb # input_b: [batch_size, seq_len, 3]

                # if dl.dataset.peak_features is not None:
                #     input_b = tc.concatenate([input_b, xb_peak], dim=-1) # input_b: [batch_size, seq_len, 3+num_peak_features]
                xb_pred = model.forward(x=input_b.swapdims(1,0), y=xb_peak.swapdims(1,0))
                xb_preds[sb,:,:] = xb_pred
                lb_sub = loss(xb_pred, xb_tgt)
                lb += lb_sub
            lb = lb/Nsubbatches
            x_preds = xb_preds[:-1,:,:].mean(dim=0)
            pred = tc.concatenate([pred, x_preds.detach().to(model.device)])

        eval_loss += lb.detach().item()

    if dl.dataset.data_incycle is None:
        assert (pred.shape[0] + xb.shape[0]) == dl.dataset.data.shape[0], AssertionError(
            "Length of prediction should match the test dataset!")
    else:
        assert (pred.shape[0] + xb[0].shape[1]) == dl.dataset.data.shape[0], AssertionError(
            "Length of prediction should match the test dataset!")

    return eval_loss/(idx+1), pred


def train_model(dl_train:tc.utils.data.DataLoader, dl_test:tc.utils.data.DataLoader,
                model:RNNiSLR, loss:tc.nn.modules.loss, optim:tc.optim,
                Nepoch, train_losses:list=None, eval_losses:list=None):
    train_losses = [] if train_losses is None else train_losses # tc.tensor([], device=model.device)
    eval_losses = [] if eval_losses is None else eval_losses # tc.tensor([], device=model.device)
    preds = tc.tensor([], device=model.device) # if preds is None else preds
    for epoch in (epoch_iter:=tqdm(range(Nepoch), leave=True, position=1)):
        # epoch_iter.set_description("Training epoch {}/{}".format(epoch, Nepoch), refresh=True)
        train_loss_in_epoch = train_one_epoch(dl_train, model, loss, optim)
        train_losses.append(train_loss_in_epoch) # = tc.concatenate([train_losses, train_loss_in_epoch])

        # epoch_iter.set_description("Evaluating epoch {}/{}".format(epoch, Nepoch), refresh=True)
        eval_loss_in_epoch, x_pred = eval_model(dl_test, model, loss)
        eval_losses.append(eval_loss_in_epoch) # = tc.concatenate([eval_losses, eval_loss_in_epoch])
        preds = tc.concatenate([preds, x_pred])

        epoch_iter.set_description("Training epoch {}/{}, train loss = {:.4f}, eval loss = {:.4f}".format(epoch, Nepoch, train_loss_in_epoch, eval_loss_in_epoch))

    return train_losses, eval_losses, preds

