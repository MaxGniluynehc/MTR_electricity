import numpy as np
import torch as tc
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression

def train_test_split(data:np.array, test_prop, seq_len):
    idx = int(data.shape[0] * test_prop)
    return data[:-idx, :], data[-idx-seq_len:,:]

class MTRiSLRDataset(Dataset):
    def __init__(self, data, data_source, seq_len, data_incycle=None, peak_features=None):
        self.data = data # [data_len, 3]
        self.colnames = ["intercept", "coef", "seq_len"]
        assert data_source in ["KBD", "TIS"], ValueError("Wrong datasource, can only be KBD or TIS!")
        self.datasource = data_source
        self.seq_len = seq_len
        self.data_incycle = data_incycle
        self.peak_features = peak_features
        if self.data_incycle is not None:
            assert self.data.shape[0] == self.data_incycle.shape[0], ValueError("len of data_incycle has to match with data!")

        if self.peak_features is not None:
            assert self.data.shape[0] == self.peak_features.shape[0], ValueError("len of peak_features has to match with data!")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.data_incycle is not None:
            if self.peak_features is not None:
                return self.data[idx:(idx+self.seq_len),:], self.data_incycle[(idx+self.seq_len)-1,:], self.peak_features[idx:(idx+self.seq_len),:]
            else:
                return self.data[idx:(idx + self.seq_len), :], self.data_incycle[(idx + self.seq_len) - 1, :], tc.empty(self.data[idx:(idx + self.seq_len), :].size())
        else:
            return self.data[idx:(idx + self.seq_len),:]

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
