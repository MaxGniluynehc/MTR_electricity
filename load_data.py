import pandas as pd
import numpy as np
import torch as tc
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from datetime import datetime

data_PATH = "/Users/maxchen/Documents/Working/ASTRI——Summer2024/MTR_electricity/dataset"

pickle_df = False
if pickle_df:
    with open(data_PATH+"/mva_2022_23_target.pkl", "wb") as file:
        df = pd.read_csv(data_PATH + "/mva_2022_23_target.csv")
        pickle.dump(df, file)

    with open(data_PATH+"/mva_2022_23.pkl", "wb") as file:
        df = pd.read_csv(data_PATH + "/mva_2022_23.csv")
        pickle.dump(df, file)

problem_timestamp_TIS = [datetime(2023,3,27,2,5,20)]
include_incycle_features = True
if not os.path.exists(data_PATH + "/mva_2022_23_cleaned.csv"):
    print("Cleaning data with time-interpolation of NaNs...")
    df = pd.read_csv(data_PATH + "/mva_2022_23.csv")
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.loc[problem_timestamp_TIS, "TIS_MAXDMD_CLP"] = np.NaN
    df_ = df.interpolate("time")

    for idx in [0,1]:
        i = np.argwhere((df_.values[:-1,idx] > df_.values[1:,idx])).flatten()
        ii = np.argwhere(i[1:] - i[:-1] == 1).flatten()
        df_.iloc[i[1:][ii], idx] = 0
    df_.iloc[0,0] = 100
    test_df_clearness = True
    if test_df_clearness:
        for idx in [0, 1]:
            i = np.argwhere((df_.values[:-1, idx] > df_.values[1:, idx])).flatten()
            ii = np.argwhere(i[1:] - i[:-1] == 1).flatten()
            assert len(ii) == 0, AssertionError("Data clearning error! Some nans are not properly reset")

        if not pd.isna(df_.values).all():
            ValueError("df_ contains NaNs!")

    if include_incycle_features:
        print("Cleaning data and adding incycle features...")
        import holidays

        hk_holidays = holidays.country_holidays('HK')

        def check_if_on_peak(dt):
            if dt.weekday() >= 6:
                return 0
            elif dt in hk_holidays:
                return 0
            elif dt.hour < 9 or dt.hour > 21:
                return 0
            else:
                return 1

        def check_rush_idle(row):
            rush_idle_ind = 0
            if row['on_peak_ind'] == 1 and row['dt'].hour >= 9 and row['dt'].hour <= 10:
                rush_idle_ind = 1
            elif row['on_peak_ind'] == 1 and row['dt'].hour >= 17 and row['dt'].hour <= 20:
                rush_idle_ind = 1
            elif row['on_peak_ind'] == 0 and row['dt'].hour >= 7 and row['dt'].hour <= 9:
                rush_idle_ind = 1
            return rush_idle_ind

        df_["dt"] = pd.to_datetime(df_.index.values)
        df_['on_peak_ind'] = df_['dt'].apply(check_if_on_peak)
        df_['rush_idle_ind'] = df_.apply(check_rush_idle, axis=1)
        df_['hour'] = pd.to_datetime(df_.dt.values).hour
    df_.to_csv(data_PATH + "/mva_2022_23_cleaned.csv")
else:
    df_ = pd.read_csv(data_PATH + "/mva_2022_23_cleaned.csv", index_col="time")
    df_.index = pd.to_datetime(df_.index)

# os.remove(data_PATH + "/mva_2022_23_cleaned.csv")

df_tgt = pd.read_csv(data_PATH+"/mva_2022_23_target.csv")
df_tgt = df_tgt.set_index("cycle")
df_tgt.index = pd.to_datetime(df_tgt.index)
df_tgt = df_tgt.sort_index()
df_tgt_ = df_tgt.interpolate("time")


# ========================== Add In-Cycle Features =========================== #
# import holidays
# hk_holidays = holidays.country_holidays('HK')
#
#
# def check_if_on_peak(dt):
#     if dt.weekday() >= 6:
#         return 0
#     elif dt in hk_holidays:
#         return 0
#     elif dt.hour < 9 or dt.hour > 21:
#         return 0
#     else:
#         return 1
#
#
# def check_rush_idle(row):
#     rush_idle_ind = 0
#     if row['on_peak_ind'] == 1 and row['dt'].hour >= 9 and row['dt'].hour <= 10:
#         rush_idle_ind = 1
#     elif row['on_peak_ind'] == 1 and row['dt'].hour >= 17 and row['dt'].hour <= 20:
#         rush_idle_ind = 1
#     elif row['on_peak_ind'] == 0 and row['dt'].hour >= 7 and row['dt'].hour <= 9:
#         rush_idle_ind = 1
#     return rush_idle_ind
#
#
# df_["dt"] = pd.to_datetime(df_.index.values)
# df_['on_peak_ind'] = df_['dt'].apply(check_if_on_peak)
# df_['rush_idle_ind'] = df_.apply(check_rush_idle, axis=1)
# df_['hour'] = pd.to_datetime(df_.dt).hour



# ================================== EDA ================================== #

EDA = False
if EDA:
    import os
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use("TkAgg")

    os.makedirs("plots/EDA", exist_ok=True)

    fig,ax = plt.subplots(2,1)
    ax[0].plot(df["KBD_MAXDMD_CLP"][:167], label="KBD_MAXDMD_CLP")
    ax[1].plot(df["TIS_MAXDMD_CLP"][:167], label="TIS_MAXDMD_CLP")
    ax[0].legend()
    ax[1].legend()
    fig.savefig("plots/EDA/mva_0-10k_signals")

    fig,ax = plt.subplots(1,1)
    ax.plot(df_["KBD_MAXDMD_CLP"][:10000], label="KBD_MAXDMD_CLP")
    ax.plot(df_["TIS_MAXDMD_CLP"][:10000], label="TIS_MAXDMD_CLP")
    ax.plot(df_tgt_.loc[:df_.index[10000],], label="target")
    ax.legend()
    fig.savefig("plots/EDA/mva_0-10k_with_target")

    # pd.DataFrame.groupby()
    # df_["KBD_MAXDMD_CLP"][0:180]
    # df_["KBD_MAXDMD_CLP"][167]
    # df_["KBD_MAXDMD_CLP"][167:167+15]
    # df_["KBD_MAXDMD_CLP"][167:167*2]
    # df_["TIS_MAXDMD_CLP"][:167]
    # df_["TIS_MAXDMD_CLP"][167:167*2]

    fig,ax = plt.subplots(1,1)
    ax.plot(df_["KBD_MAXDMD_CLP"][:10000] + df_["TIS_MAXDMD_CLP"][:10000], label="KBD+TIS")
    # ax.plot(df_["TIS_MAXDMD_CLP"][:2000], label="TIS_MAXDMD_CLP")
    ax.plot(df_tgt_.loc[:df_.index[10000],], label="target")
    ax.legend()
    fig.savefig("plots/EDA/mva_agg_0-10k_with_target")


    fig,ax = plt.subplots(1,1)
    ax.plot(df_tgt_.values[:500], label="target")
    ax.legend()
    fig.savefig("plots/EDA/mva_target_0-5k")


# ====================== Divide dataset in to subgroups ========================== #
from sklearn.linear_model import LinearRegression

# fig, ax = plt.subplots(1,1)
# for i in range(len(subseqs)):
#     ax.plot(subseqs[i], color="gray")

# s = subseqs[0].reshape([-1, 1])
# x = np.linspace(1,len(s), num=len(s)).reshape([-1, 1])
# lm = LinearRegression().fit(x,s)
#
# fig, ax = plt.subplots(1,1)
# ax.plot(x,s, linestyle="none", marker="o", ms=3, color="gray")
# ax.plot(x, lm.predict(x).flatten(), color="red")


def fit_iterative_SLR(signal, num_subintervals:int|None =3):
    partitions = np.concatenate([np.array([-1]), np.argwhere(signal[1:] < signal[:-1]).flatten()])
    subseqs = [signal[(partitions[p]+1):partitions[p+1]] for p in range(len(partitions)-1)]
    subseqs.append(signal[(partitions[-1]+1):])
    assert all([seq[-1] > seq[0] for seq in subseqs]), AssertionError("Wrong partition of the signals!")

    iter_lms = np.zeros([len(subseqs), 3]) # 3dims: intercept, coef(slope), step_len
    iter_lms_subs = None
    for idx, s in enumerate(subseqs):
        s = s.reshape([-1, 1])
        x = np.linspace(1, len(s), num=len(s)).reshape([-1, 1])
        lm = LinearRegression().fit(x, s)
        iter_lms[idx, :] = np.concatenate([lm.intercept_, lm.coef_[0], np.array([len(s)])])

    if num_subintervals is not None:
        iter_lms_subs = np.zeros([len(subseqs), 3*(num_subintervals-1)])
        for idx, s in enumerate(subseqs):
            s = s.reshape([-1, 1])
            cutoff_idx = np.floor(np.arange(1,num_subintervals)/num_subintervals * s.shape[0]).astype(dtype=int)
            for i_sub, c_idx in enumerate(cutoff_idx):
                s_sub = s[:c_idx, :]
                x_sub = np.linspace(1, len(s_sub), num=len(s_sub)).reshape([-1,1])
                lm_sub = LinearRegression().fit(x_sub, s_sub)
                iter_lms_subs[idx, (i_sub*num_subintervals):(i_sub*num_subintervals+3)] = np.concatenate([lm_sub.intercept_, lm_sub.coef_[0], np.array([len(s_sub)])])

    return iter_lms, iter_lms_subs


if any([(not os.path.exists(data_PATH+"/iter_lms_KBD")),
        (not os.path.exists(data_PATH+"/iter_lms_TIS")),
        (not os.path.exists(data_PATH+"/iter_lms_KBD_subs")),
        (not os.path.exists(data_PATH+"/iter_lms_TIS_subs"))]):
    print("(Re)Loading data via iterative Simple Linear Regressions...")
    signals = df_[["KBD_MAXDMD_CLP", "TIS_MAXDMD_CLP"]].values  # df_.values
    iter_lms_KBD, iter_lms_KBD_subs = fit_iterative_SLR(signals[:,0])
    iter_lms_TIS, iter_lms_TIS_subs = fit_iterative_SLR(signals[:,1])
    with open(data_PATH+"/iter_lms_KBD", "wb") as f1:
        pickle.dump(iter_lms_KBD, f1)
    with open(data_PATH+"/iter_lms_TIS", "wb") as f2:
        pickle.dump(iter_lms_TIS, f2)
    with open(data_PATH+"/iter_lms_KBD_subs", "wb") as f1s:
        pickle.dump(iter_lms_KBD_subs, f1s)
    with open(data_PATH+"/iter_lms_TIS_subs", "wb") as f2s:
        pickle.dump(iter_lms_TIS_subs, f2s)
else:
    with open(data_PATH+"/iter_lms_KBD", "rb") as f1:
        iter_lms_KBD = pickle.load(f1)
    with open(data_PATH+"/iter_lms_TIS", "rb") as f2:
        iter_lms_TIS = pickle.load(f2)
    with open(data_PATH+"/iter_lms_KBD_subs", "rb") as f1s:
        iter_lms_KBD_subs = pickle.load(f1s)
    with open(data_PATH+"/iter_lms_TIS_subs", "rb") as f2s:
        iter_lms_TIS_subs = pickle.load(f2s)

# iter_lms_KBD.shape
# iter_lms_TIS.shape
# df_tgt_.shape
#
#
# df.iloc[896:910,]
# df_.iloc[896:910,]
#
# df.iloc[1250:1260,]
# df_.iloc[1250:1260,]
#
# df_v = df_.values
#
# np.argwhere(df_v[1251,] == 0)[0]
# np.argwhere(df_v[1250,] == 0)
#
# for r in range(1, df_v.shape[0]):
#     if len(np.argwhere(df_v[r,] == 0)) == 1:
#         c = np.argwhere(df_v[r,] == 0)[0]
#         c_ = np.abs(1-c)
#         if df_v[r,c_] >= df_v[r-1,c_]:
#             df_v[r,c] = df_v[r-1,c]
#
#
# pt1 = np.concatenate([np.array([-1]), np.argwhere(df_v[1:,0] < df_v[:-1, 0]).flatten()])
# pt2 = np.concatenate([np.array([-1]), np.argwhere(df_v[1:,1] < df_v[:-1, 1]).flatten()])
# len(pt1)
# len(pt2)
#
# for i in range(1798, len(pt1)):
#     if pt1[i] != pt2[i]:
#         print(i)
#         break
# pt1[1790:1800]
# pt2[1790:1800]
# #
# df_.iloc[321993:322010,]  # TIS has more wave than KBD?
# df.iloc[321993:322010,]  # TIS has more wave than KBD?
#
# df_v[321993:322010,]


def train_test_split(data:np.array, test_prop, seq_len):
    idx = int(data.shape[0] * test_prop)
    return data[:-idx, :], data[-idx-seq_len:,:]


class MTRiSLRDataset(Dataset):
    def __init__(self, data, data_source, seq_len, data_incycle=None):
        self.data = data # [data_len, 3]
        self.colnames = ["intercept", "coef", "seq_len"]
        assert data_source in ["KBD", "TIS"], ValueError("Wrong datasource, can only be KBD or TIS!")
        self.datasource = data_source
        self.seq_len = seq_len
        self.data_incycle = data_incycle
        if self.data_incycle is not None:
            assert self.data.shape[0] == self.data_incycle.shape[0], ValueError("len of data_incycle has to match with data!")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.data_incycle is not None:
            return self.data[idx:(idx+self.seq_len),:], self.data_incycle[(idx+self.seq_len)-1,:]
        else:
            return self.data[idx:(idx + self.seq_len),:]



if __name__ == '__main__':
    from datetime import datetime
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("TkAgg")
    start_date = df_.index[0]
    end_date = datetime(2022,2,6,23,59,30)
    signals = df_.loc[start_date:end_date,].values
    iter_lms_KBD = fit_iterative_SLR(signals[:,0])
    iter_lms_TIS = fit_iterative_SLR(signals[:,1])

    pred_KBD = iter_lms_KBD[:,0] + iter_lms_KBD[:,1] * iter_lms_KBD[:,2]
    pred_TIS = iter_lms_TIS[:,0] + iter_lms_TIS[:,1] * iter_lms_TIS[:,2]

    pred_KBD.shape
    pred_TIS.shape

    pred_tgt = pred_KBD + pred_TIS
    pred_tgt.shape

    tgt = df_tgt_.loc[start_date:end_date,].values
    fig,ax=plt.subplots(1,1)
    ax.plot(tgt,label="tgt")
    ax.plot(pred_tgt, label="pred_tgt")
    ax.legend()
    fig.savefig("plots/EDA/iterative_SLR")

