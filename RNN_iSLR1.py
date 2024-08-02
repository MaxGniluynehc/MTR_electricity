import pandas as pd
import numpy as np
import torch as tc
from torch.nn import Module, LSTM, MultiheadAttention, Linear, Embedding, Sequential, Dropout, ReLU, Tanh

'''
Used for pilot6+
'''


class RNNiSLR(Module):
    """
    data: [seq_len, batch_size, 3], 3 corresponds to [intercept, coef, step_size]

    """
    def __init__(self, in_size, num_hidden, hidden_dim, n_features=3, n_yfeatures=None, use_attention=False, device=None):
        super(RNNiSLR, self).__init__()
        if device is None and tc.cuda.is_available():
            self.device = "cuda: 2"
        elif device is None and tc.backends.mps.is_available():
            self.device = "mps"
        elif device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.dropout_p = [0.3, 0.3] if self.training else [0,0]
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.in_size = in_size
        self.fc_inx1 = Linear(self.n_features, self.in_size, device=self.device)
        self.fc_inx2 = Linear(self.in_size, self.in_size, device=self.device)

        if n_yfeatures is not None:
            self.n_yfeatures = n_yfeatures
            self.fc_iny1 = Linear(self.n_yfeatures, self.in_size, bias=False, device=self.device)
            self.fc_iny2 = Linear(self.in_size, 1, bias=False, device=self.device)
        self.lstm_enc = LSTM(input_size=self.in_size, hidden_size=hidden_dim, num_layers=num_hidden, dropout=self.dropout_p[1], device=self.device)

        self.use_attention = use_attention
        if self.use_attention:
            self.att_q = Linear(self.in_size, hidden_dim, device=self.device)
            self.att_k = Linear(self.in_size, hidden_dim, device=self.device)
            self.att_v = Linear(hidden_dim, hidden_dim, device=self.device)
            self.att = MultiheadAttention(hidden_dim, num_hidden, device=self.device)

        self.lstm_dec = LSTM(input_size=hidden_dim, hidden_size=self.in_size, num_layers=num_hidden, dropout=self.dropout_p[1], device=self.device)
        self.fc_out = Linear(self.in_size, 3, device=self.device)

    def initialize_lstm(self, batch_size, init_hidden=True):
        if init_hidden:
            h0, c0 = tc.zeros([self.num_hidden, batch_size, self.hidden_dim], device=self.device), tc.zeros(
                [self.num_hidden, batch_size, self.hidden_dim], device=self.device)
            return (h0, c0)
        else:
            return None

    def forward(self, x, y=None, hidden=None):
        # x: [seq_len, batch_size, 3]
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        batch_size = x.shape[1]
        if any([hidden is None]):
            hidden = self.initialize_lstm(batch_size=batch_size)

        x_in = Sequential(self.fc_inx1, Dropout(self.dropout_p[0]), Tanh())(x) if self.training else Sequential(self.fc_inx1, Tanh())(x) # x_in: [seq_len, batch_size, in_size]
        if y is not None:
            y_in = Sequential(self.fc_iny1, Dropout(self.dropout_p[0]), Tanh(), self.fc_iny2, Dropout(self.dropout_p[0]), Tanh())(y) if self.training \
                else Sequential(self.fc_iny1, Tanh(), self.fc_iny2, Tanh())(y)
            x_in = x_in.add(y_in.repeat([1,1,self.in_size]))
        x_in = Sequential(self.fc_inx2, Dropout(self.dropout_p[0]), Tanh())(x_in) if self.training else Sequential(self.fc_inx2, Tanh())(x_in)
        state, (hn, cn) = self.lstm_enc(x_in, hidden) # state: [seq_len, batch_size, hidden_dim]

        if self.use_attention:
            x_q = self.att_q(x_in)
            x_k = self.att_k(x_in)
            z_v = self.att_v(state)
            state,_ = self.att(x_q, x_k, z_v, need_weights=False) # after attention, state: [seq_len, batch_size, hidden_dim]

        x_dec , _ = self.lstm_dec(state, (hn, cn)) # x_dec: [seq_len, batch_size, in_size]
        x_dec = x_dec.swapdims(1,0).mean(1) # x_dec: [batch_size, in_size]
        out = Sequential(self.fc_out, Dropout(self.dropout_p[0]))(x_dec) if self.training else self.fc_out(x_dec) # out: [batch_size, 3]

        return out

