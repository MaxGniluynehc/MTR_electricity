import os
print("Loading model from RNN_iSLR1 !!!")
from RNN_iSLR1 import RNNiSLR
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
import argparse

if True:
    parser = argparse.ArgumentParser(prog="pilot_train_parser")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--pilot_dataset_size", type=int, default=5000)
    parser.add_argument("--test_prop", type=float, default=0.3)
    parser.add_argument("--loss_fn_names", nargs="+", default=["l1_loss", "mse_loss", "huber_loss"])
    parser.add_argument("--channel_weights", nargs="+", type=float, default=[0.4, 0.4, 0.2])
    parser.add_argument("--train_batch_size", type=int, default=15)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--Nepoch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot_name")
    parser.add_argument("--model_name")
    parser.add_argument("--continuous_training", action="store_true")
    parser.add_argument("--train_self", action="store_true")
    parser.add_argument("--save_trained", action="store_true")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()


# seq_len = 5
KBD_pilot = iter_lms_KBD[:args.pilot_dataset_size,:]
KBD_subs_pilot = iter_lms_KBD_subs[:args.pilot_dataset_size,:]
KBD_peaks_pilot = peak_features_KBD[:args.pilot_dataset_size, :]

KBD_train, KBD_test = train_test_split(KBD_pilot, args.test_prop, args.seq_len)
KBD_subs_train, KBD_subs_test = train_test_split(KBD_subs_pilot, args.test_prop, args.seq_len)
KBD_peaks_train, KBD_peaks_test = train_test_split(KBD_peaks_pilot, args.test_prop, args.seq_len)

scaler = StandardScaler()
KBD_train_sc, KBD_test_sc = scaler.fit_transform(KBD_train), scaler.fit_transform(KBD_test)

scaler_peaks = StandardScaler()
KBD_peaks_train_sc, KBD_peaks_test_sc = scaler_peaks.fit_transform(KBD_peaks_train), scaler_peaks.fit_transform(KBD_peaks_test)

scaler_subs = StandardScaler()
KBD_subs_train_sc, KBD_subs_test_sc = scaler_subs.fit_transform(KBD_subs_train), scaler_subs.fit_transform(KBD_subs_test)

# ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", args.seq_len)
ds_train = MTRiSLRDataset(tc.tensor(KBD_train_sc, dtype=tc.float32), "KBD", args.seq_len,
                          data_incycle=tc.tensor(KBD_subs_train_sc, dtype=tc.float32),
                          peak_features=tc.tensor(KBD_peaks_train_sc, dtype=tc.float32))
# ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", args.seq_len)
ds_test = MTRiSLRDataset(tc.tensor(KBD_test_sc, dtype=tc.float32), "KBD", args.seq_len,
                         data_incycle=tc.tensor(KBD_subs_test_sc, dtype=tc.float32),
                         peak_features=tc.tensor(KBD_peaks_test_sc, dtype=tc.float32))
dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=False, drop_last=False)
dl_test = DataLoader(ds_test, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

# pilot_name = "pilot6_1"
# model_name = "_att_m4m4h2"
# continuous_training = True

self = RNNiSLR(6, 3, 6, 3,3, True, device=args.device)
if args.continuous_training:
    print("Loading saved model for continuous training...")
    self.load_state_dict(tc.load("checkpoints/{}/{}{}".format(args.pilot_name, args.pilot_name, args.model_name), map_location=args.device))
loss = TraininngLoss(name="l_m_h", channel_weights=tc.tensor(args.channel_weights), loss_fn_names=args.loss_fn_names)
optim = Adam(self.parameters(), lr=args.lr, betas=(0.9, 0.999))


# train_self = False
if args.train_self:
    s = time.time()
    if not args.continuous_training:
        train_losses, eval_losses, preds = None, None, None
    else:
        print("Loading saved train_losses and eval_losses for continuous training...")
        train_losses = np.fromfile("checkpoints/{}/{}{}_train_losses".format(args.pilot_name, args.pilot_name, args.model_name)).tolist()
        eval_losses = np.fromfile("checkpoints/{}/{}{}_eval_losses".format(args.pilot_name, args.pilot_name, args.model_name)).tolist()

    train_losses, eval_losses, preds = train_model(dl_train, dl_test, self, loss, optim, args.Nepoch,
                                                   train_losses, eval_losses)
    print("\n Time elasped: {} min".format((time.time() - s) / 60))


# save_trained = False
if args.save_trained:
    os.makedirs("checkpoints/{}/".format(args.pilot_name), exist_ok=True)
    tc.save(self.state_dict(), "checkpoints/{}/{}{}".format(args.pilot_name, args.pilot_name, args.model_name))
    tc.save(preds, "checkpoints/{}/{}{}_preds".format(args.pilot_name, args.pilot_name, args.model_name))
    np.array(train_losses).tofile("checkpoints/{}/{}{}_train_losses".format(args.pilot_name, args.pilot_name, args.model_name))
    np.array(eval_losses).tofile("checkpoints/{}/{}{}_eval_losses".format(args.pilot_name, args.pilot_name, args.model_name))

# plot_results=True
# save_plots=True
if args.plot_results:
    print("Generating plots ....")
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    # matplotlib.use("TkAgg")

    os.makedirs("plots/{}/".format(args.pilot_name), exist_ok=True)

    train_losses = np.fromfile("checkpoints/{}/{}{}_train_losses".format(args.pilot_name, args.pilot_name, args.model_name))
    eval_losses = np.fromfile("checkpoints/{}/{}{}_eval_losses".format(args.pilot_name, args.pilot_name, args.model_name))

    fig1,ax = plt.subplots(1,1)
    ax.plot(train_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(args.model_name),colors="red")
    ax.set_title("Training Loss")
    # ax.legend()
    fig1.show()

    fig2,ax=plt.subplots()
    ax.plot(eval_losses)
    # ax.vlines(300, ymax=0.34, ymin=0.24, label="{} start".format(args.model_name),colors="red")
    ax.set_title("Evaluation Loss")
    # ax.legend()
    fig2.show()

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

    fig3,ax = plt.subplots(1,1)
    ax.plot(KBD_test_toplot, label="target")
    ax.plot(x_preds_toplot, label="pred")
    ax.legend()
    ax.set_title("Aggregated")
    fig3.show()

    fig4,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,1], label="target")
    ax.plot(x_preds[:,1], label="pred")
    ax.legend()
    ax.set_title("Slope")
    fig4.show()

    fig5,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,0], label="target")
    ax.plot(x_preds[:,0], label="pred")
    ax.legend()
    ax.set_title("Intercept")
    fig5.show()

    fig6,ax = plt.subplots(1,1)
    ax.plot(KBD_test[5:,2], label="target")
    ax.plot(x_preds[:,2], label="pred")
    ax.legend()
    ax.set_ylim([150,190])
    ax.set_title("Step Size")
    fig6.show()

    if args.save_plots:
        fig1.savefig("plots/{}/{}{}_training_loss".format(args.pilot_name, args.pilot_name, args.model_name))
        fig2.savefig("plots/{}/{}{}_eval_loss".format(args.pilot_name, args.pilot_name, args.model_name))
        fig3.savefig("plots/{}/{}{}_aggregated".format(args.pilot_name, args.pilot_name, args.model_name))
        fig4.savefig("plots/{}/{}{}_slope".format(args.pilot_name, args.pilot_name, args.model_name))
        fig5.savefig("plots/{}/{}{}_intercept".format(args.pilot_name, args.pilot_name, args.model_name))
        fig6.savefig("plots/{}/{}{}_stepsize".format(args.pilot_name, args.pilot_name, args.model_name))

    del fig1, fig2, fig3, fig4, fig5, fig6, ax








