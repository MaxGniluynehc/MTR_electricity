import os
from RNN_iSLR import RNNiSLR
from load_data import iter_lms_KBD, iter_lms_TIS, fit_iterative_SLR, MTRiSLRDataset, train_test_split
import torch as tc
from torch.utils.data import DataLoader
from torch.nn import MSELoss, HuberLoss
from torch.optim import Adam, AdamW
from torch.nn.functional import huber_loss, mse_loss, smooth_l1_loss, l1_loss
from tqdm import tqdm
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from train_RNNiSLR import train_model, eval_model


seq_len = 5
KBD_pilot = iter_lms_KBD[:5000,:]

KBD_train, KBD_test = train_test_split(KBD_pilot, 0.3, seq_len)

scaler = StandardScaler()
KBD_train_sc = scaler.fit_transform(KBD_train)
KBD_test_sc = scaler.fit_transform(KBD_test)

ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", seq_len)
ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", seq_len)
dl_train = DataLoader(ds_train, batch_size=15, shuffle=False, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=10, shuffle=False, drop_last=True)
# len(list(dl_train))
# len(list(dl_test))
# next(iter(dl_train)).shape


class TraininngLoss(tc.nn.Module):
    def __init__(self, reduction="mean", channel_weights = tc.ones(3)/3, name=None):
        super().__init__()
        self.reduction = reduction
        self.channel_weights = channel_weights if channel_weights.sum()== 1 else channel_weights/channel_weights.sum()
        self.name = name

    def forward(self, input:tc.Tensor, target:tc.Tensor)-> tc.Tensor:
        loss_intercept = mse_loss(input[:,0], target[:,0], reduction=self.reduction)
        loss_coef = mse_loss(input[:,1], target[:,1], reduction=self.reduction)
        loss_stepsize = huber_loss(input[:,2], target[:,2], reduction=self.reduction)
        return loss_intercept*self.channel_weights[0] + loss_coef*self.channel_weights[1] + loss_stepsize*self.channel_weights[2]
        # tc.tensor([loss_intercept, loss_coef, loss_stepsize], requires_grad=True).matmul(self.channel_weights)


self = RNNiSLR(6, 3, 6, True, device="mps")
self.load_state_dict()
loss = TraininngLoss(name="m_m_m", channel_weights=tc.tensor([0.4, 0.4, 0.2]))
optim = Adam(self.parameters(), lr=1*1e-4, betas=(0.9, 0.999))


s = time.time()
train_losses, eval_losses, preds = None, None, None
train_losses, eval_losses, preds = train_model(dl_train, dl_test, self, loss, optim, 500,
                                               train_losses, eval_losses)
print("Time elasped: {} min".format((time.time() - s) / 60))

pilot_name = "pilot2"
model_name = "_att_m4m4m2"

save_trained = False
if save_trained:
    os.makedirs("checkpoints/{}/".format(pilot_name), exist_ok=True)
    tc.save(self.state_dict(), "checkpoints/{}/{}{}".format(pilot_name, pilot_name, model_name))
    tc.save(preds, "checkpoints/{}/{}{}_preds".format(pilot_name, pilot_name, model_name))
    np.array(train_losses).tofile("checkpoints/{}/{}{}_train_losses".format(pilot_name, pilot_name, model_name))
    np.array(eval_losses).tofile("checkpoints/{}/{}{}_eval_losses".format(pilot_name, pilot_name, model_name))
np.array([2,3,4]).tolist()

plot_results=False
if plot_results:
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use("TkAgg")

    os.makedirs("plots/{}/".format(pilot_name), exist_ok=True)

    fig,ax=plt.subplots()
    ax.plot(train_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(model_name),colors="red")
    ax.set_title("Training Loss")
    ax.legend()
    fig.show()
    fig.savefig("plots/{}/{}_training_loss".format(pilot_name, pilot_name))

    fig,ax=plt.subplots()
    ax.plot(eval_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(model_name),colors="red")
    ax.set_title("Evaluation Loss")
    ax.legend()
    fig.show()
    fig.savefig("plots/{}/{}_eval_loss".format(pilot_name, pilot_name))

    _, x_preds_sc = eval_model(dl_test, self, loss)
    # x_preds_sc.shape
    # KBD_test_sc.shape
    x_preds = scaler.inverse_transform(x_preds_sc.cpu())
    x_preds_toplot = x_preds[:,0] + x_preds[:,1] * x_preds[:,2]
    KBD_test_toplot = KBD_test[5:,0] + KBD_test[5:,1] * KBD_test[5:,2]

    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(x_preds_toplot, label="pred")
    # ax[1].plot(KBD_test_toplot, label="target")
    # ax[0].legend()
    # ax[1].legend()
    # fig.show()

    fig,ax = plt.subplots(1,1)
    ax.plot(KBD_test_toplot, label="target")
    ax.plot(x_preds_toplot, label="pred")
    ax.legend()
    ax.set_title("Aggregated")
    fig.show()
    fig.savefig("plots/{}/{}_aggregated".format(pilot_name, pilot_name))

    fig,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,1], label="target")
    ax.plot(x_preds[:,1], label="pred")
    ax.legend()
    ax.set_title("Slope")
    fig.show()
    fig.savefig("plots/{}/{}_slope".format(pilot_name,pilot_name))

    fig,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,0], label="target")
    ax.plot(x_preds[:,0], label="pred")
    ax.legend()
    ax.set_title("Intercept")
    fig.show()
    fig.savefig("plots/{}/{}_intercept".format(pilot_name,pilot_name))

    fig,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,2], label="target")
    ax.plot(x_preds[:,2], label="pred")
    ax.legend()
    ax.set_ylim([150,190])
    ax.set_title("Step Size")
    fig.show()
    fig.savefig("plots/{}/{}_stepsize".format(pilot_name, pilot_name))




