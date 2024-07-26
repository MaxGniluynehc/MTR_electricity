import os
from RNN_iSLR import RNNiSLR
from load_data import iter_lms_KBD, iter_lms_TIS, fit_iterative_SLR, MTRiSLRDataset, train_test_split
import torch as tc
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import time


# seq_len = 5
# KBD_pilot = iter_lms_KBD[:5000,:]
# KBD_subs_pilot = iter_lms_KBD_subs[:5000,:]
#
# KBD_train, KBD_test = train_test_split(KBD_pilot, 0.3, seq_len)
# KBD_subs_train, _ = train_test_split(KBD_subs_pilot, 0.3, seq_len)
#
# scaler = StandardScaler()
# KBD_train_sc = scaler.fit_transform(KBD_train)
# KBD_subs_train_sc = scaler.fit_transform(KBD_subs_train)
# # KBD_test_sc = scaler.fit_transform(KBD_test)
#
# ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", seq_len, data_incycle=tc.tensor(KBD_subs_train_sc, dtype=tc.float32))
# # ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", seq_len)
# dl_train = DataLoader(ds_train, batch_size=15, shuffle=False, drop_last=True)
# dl_test = DataLoader(ds_test, batch_size=10, shuffle=False, drop_last=True)

def train_one_epoch(dl:tc.utils.data.DataLoader, model:RNNiSLR, loss:tc.nn.modules.loss, optim:tc.optim):
    loss_in_epoch = 0
    model.train(True)

    if dl.dataset.data_incycle is not None:
        _, xb_sub = next(iter(dl))
        Nsubbatches = int(xb_sub.size(-1) / 3) + 2
        num_subintervals = int(xb_sub.size(-1) / 3) + 1
    else:
        Nsubbatches, num_subintervals = 0,0

    for idx, xb in enumerate(dl):
        # dl_iter.set_description("Training on epoch...")
        if dl.dataset.data_incycle is None:
            xb = xb.swapdims(1,0).to(model.device)
            optim.zero_grad()
            lb = loss(model.forward(xb[:-1, :, :]), xb[-1, :, :])
            lb.backward()
            optim.step()
        else:
            xbb, xb_sub = xb # xbb: [batch_size, seq_len, 3], xb_sub: [batch_size, 3*num_subintervals]
            xbb, xb_sub = xbb.to(model.device), xb_sub.to(model.device)
            xb_tgt = xbb[:,-1,:] # xb_tgt: [batch_size, 3]
            # Nsubbatches = int(xb_sub.size(-1)/3) + 2
            # num_subintervals = int(xb_sub.size(-1)/3) + 1
            lb=tc.zeros(1).to(model.device)
            for sb in range(Nsubbatches):
                if sb == 0:
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=",xbb[:,:-1,:].shape)
                    input_b = tc.concatenate([xbb[:, :-1, :], tc.zeros([xbb.size(0), 1, 3])], dim=1)
                elif (sb > 0) and (sb < (Nsubbatches-1)):
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=", xbb[:, :-1, :].shape)
                    # print("xb_sub slice shape=", xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3).shape)
                    input_b = tc.concatenate([xbb[:, :-1, :],
                                              xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3)], dim=1)
                else:
                    input_b = xbb # input_b: [batch_size, seq_len, 3]
                optim.zero_grad()
                lb_sub = loss(model.forward(input_b.swapdims(1,0)), xb_tgt)
                lb_sub.backward()
                optim.step()
                lb += lb_sub
            lb = lb/Nsubbatches
        loss_in_epoch += lb.detach().item()
    return loss_in_epoch/(idx+1)


def eval_model(dl:tc.utils.data.DataLoader, model:RNNiSLR, loss:tc.nn.modules.loss):
    eval_loss = 0
    model.train(False)
    pred = tc.tensor([], device=model.device)

    if dl.dataset.data_incycle is not None:
        _, xb_sub = next(iter(dl))
        Nsubbatches = int(xb_sub.size(-1) / 3) + 2
        num_subintervals = int(xb_sub.size(-1) / 3) + 1
    else:
        Nsubbatches, num_subintervals = 0,0

    for idx, xb in enumerate(dl):
        # dl_iter.set_description("Evaluating on epoch...")
        if dl.dataset.data_incycle is None:
            xb = xb.swapdims(1,0).to(model.device)
            x_pred = model.forward(xb[:-1, :, :])
            pred = tc.concatenate([pred, x_pred.detach()])
            lb = loss(x_pred, xb[-1, :, :])

        else:
            xbb, xb_sub = xb # xbb: [batch_size, seq_len, 3], xb_sub: [batch_size, 3*num_subintervals]
            xbb, xb_sub = xbb.to(model.device), xb_sub.to(model.device)
            xb_tgt = xbb[:,-1,:] # xb_tgt: [batch_size, 3]
            # Nsubbatches = int(xb_sub.size(-1)/3) + 2
            # num_subintervals = int(xb_sub.size(-1)/3) + 1
            lb=tc.zeros(1).to(model.device)
            xb_preds = tc.zeros([Nsubbatches, xb_tgt.size(0), 3])
            for sb in range(Nsubbatches):
                if sb == 0:
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=",xbb[:,:-1,:].shape)
                    input_b = tc.concatenate([xbb[:, :-1, :], tc.zeros([xbb.size(0), 1, 3])], dim=1)
                elif (sb > 0) and (sb < (Nsubbatches-1)):
                    # print("xbb when sb=={}: {}".format(sb, xbb.shape))
                    # print("xbb[:,:-1,:].shape=", xbb[:, :-1, :].shape)
                    # print("xb_sub slice shape=", xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3).shape)
                    input_b = tc.concatenate([xbb[:, :-1, :],
                                              xb_sub[:,(sb-1)*num_subintervals:(sb-1)*num_subintervals+3].view(-1,1,3)], dim=1)
                else:
                    input_b = xbb # input_b: [batch_size, seq_len, 3]
                xb_pred = model.forward(input_b.swapdims(1,0))
                xb_preds[sb,:,:] = xb_pred
                lb_sub = loss(xb_pred, xb_tgt)
                lb += lb_sub

            lb = lb/Nsubbatches
            x_preds = xb_preds[:-1,:,:].mean(dim=0)
            pred = tc.concatenate([pred, x_preds.detach()])

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
    for epoch in (epoch_iter:=tqdm(range(Nepoch), leave=False, position=1)):
        epoch_iter.set_description("Training epoch {}/{}".format(epoch, Nepoch), refresh=True)
        train_loss_in_epoch = train_one_epoch(dl_train, model, loss, optim)
        train_losses.append(train_loss_in_epoch) # = tc.concatenate([train_losses, train_loss_in_epoch])

        epoch_iter.set_description("Evaluating epoch {}/{}".format(epoch, Nepoch), refresh=True)
        eval_loss_in_epoch, x_pred = eval_model(dl_test, model, loss)
        eval_losses.append(eval_loss_in_epoch) # = tc.concatenate([eval_losses, eval_loss_in_epoch])
        preds = tc.concatenate([preds, x_pred])

        epoch_iter.set_description("Training epoch {}/{}, train loss = {:.4f}, eval loss = {:.4f}".format(epoch, Nepoch, train_loss_in_epoch, eval_loss_in_epoch))

    return train_losses, eval_losses, preds



# if __name__ == '__main__':
#     seq_len = 5
#     KBD_train, KBD_test = train_test_split(iter_lms_KBD, 0.3, seq_len)
#
#     ds_train = MTRiSLRDataset(tc.tensor(KBD_train, dtype=tc.float32), "KBD", seq_len)
#     ds_test = MTRiSLRDataset(tc.tensor(KBD_test, dtype=tc.float32), "KBD", seq_len)
#     dl_train = DataLoader(ds_train, batch_size=10, shuffle=False, drop_last=True)
#     dl_test = DataLoader(ds_test, batch_size=10, shuffle=False, drop_last=True)
#     # next(iter(dl)).shape
#     self = RNNiSLR(6,3,6,False)
#     loss = MSELoss()
#     optim = Adam(self.parameters(), lr=1e-3)
#
#     # train_loss = train_one_epoch(dl_train, self, loss, optim)
#     #
#     # eval_loss, x_pred = eval_model(dl_test, self, loss)
#
#     s = time.time()
#     train_losses, eval_losses, preds = train_model(dl_train, dl_test, self, loss, optim, 50)
#     print("Time elasped: {} min".format((time.time() - s)/60))
#
#     os.makedirs("checkpoints", exist_ok=True)
#     tc.save(self.state_dict(), "checkpoints/m1")
#     import argparse



