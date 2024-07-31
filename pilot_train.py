import os
from RNN_iSLR import RNNiSLR
from load_data import iter_lms_KBD, iter_lms_KBD_subs, peak_features_KBD
from data_util_fn import MTRiSLRDataset, train_test_split
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
from loss_fn import TraininngLoss

seq_len = 5
KBD_pilot = iter_lms_KBD[:5000,:]
KBD_subs_pilot = iter_lms_KBD_subs[:5000,:]
KBD_peaks_pilot = peak_features_KBD[:5000, :]

KBD_train, KBD_test = train_test_split(KBD_pilot, 0.3, seq_len)
KBD_subs_train, KBD_subs_test = train_test_split(KBD_subs_pilot, 0.3, seq_len)
KBD_peaks_train, KBD_peaks_test = train_test_split(KBD_peaks_pilot, 0.3, seq_len)

scaler = StandardScaler()
KBD_train_sc, KBD_test_sc = scaler.fit_transform(KBD_train), scaler.fit_transform(KBD_test)

scaler_peaks = StandardScaler()
KBD_peaks_train_sc, KBD_peaks_test_sc = scaler_peaks.fit_transform(KBD_peaks_train), scaler_peaks.fit_transform(KBD_peaks_test)

scaler_subs = StandardScaler()
KBD_subs_train_sc, KBD_subs_test_sc = scaler_subs.fit_transform(KBD_subs_train), scaler_subs.fit_transform(KBD_subs_test)

# ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", seq_len)
ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", seq_len,
                          data_incycle=tc.tensor(KBD_subs_train_sc, dtype=tc.float32),
                          peak_features=tc.tensor(KBD_peaks_train_sc, dtype=tc.float32))
# ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", seq_len)
ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", seq_len,
                         data_incycle=tc.tensor(KBD_subs_test_sc, dtype=tc.float32),
                         peak_features=tc.tensor(KBD_peaks_test_sc, dtype=tc.float32))
dl_train = DataLoader(ds_train, batch_size=15, shuffle=False, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=10, shuffle=False, drop_last=True)

pilot_name = "pilot6"
model_name = "_att_m4m4h2"

self = RNNiSLR(6, 3, 6, 3,3, True, device="cpu")
# self.load_state_dict(tc.load("checkpoints/{}/{}{}".format(pilot_name, pilot_name, model_name), map_location="cpu"))
loss = TraininngLoss(name="m_m_m", channel_weights=tc.tensor([0.4, 0.4, 0.2]))
optim = Adam(self.parameters(), lr=1*1e-4, betas=(0.9, 0.999))


train_self = False
if train_self:
    s = time.time()
    train_losses, eval_losses, preds = None, None, None
    train_losses, eval_losses, preds = train_model(dl_train, dl_test, self, loss, optim, 2,
                                                   train_losses, eval_losses)
    print("\n Time elasped: {} min".format((time.time() - s) / 60))


save_trained = False
if save_trained:
    os.makedirs("checkpoints/{}/".format(pilot_name), exist_ok=True)
    tc.save(self.state_dict(), "checkpoints/{}/{}{}".format(pilot_name, pilot_name, model_name))
    tc.save(preds, "checkpoints/{}/{}{}_preds".format(pilot_name, pilot_name, model_name))
    np.array(train_losses).tofile("checkpoints/{}/{}{}_train_losses".format(pilot_name, pilot_name, model_name))
    np.array(eval_losses).tofile("checkpoints/{}/{}{}_eval_losses".format(pilot_name, pilot_name, model_name))


plot_results=False
if plot_results:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    # matplotlib.use("TkAgg")

    os.makedirs("plots/{}/".format(pilot_name), exist_ok=True)

    train_losses = np.fromfile("checkpoints/{}/{}{}_train_losses".format(pilot_name, pilot_name, model_name))
    eval_losses = np.fromfile("checkpoints/{}/{}{}_eval_losses".format(pilot_name, pilot_name, model_name))

    fig,ax = plt.subplots(1,1)
    ax.plot(train_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(model_name),colors="red")
    ax.set_title("Training Loss")
    # ax.legend()
    fig.show()
    fig.savefig("plots/{}/{}_training_loss".format(pilot_name, pilot_name))

    fig,ax=plt.subplots()
    ax.plot(eval_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(model_name),colors="red")
    ax.set_title("Evaluation Loss")
    # ax.legend()
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




