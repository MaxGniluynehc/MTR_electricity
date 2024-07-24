import pandas as pd
import numpy as np
import torch as tc
from torch.nn import Module, LSTM, MultiheadAttention, Linear, Embedding, Sequential, Dropout


class RNNiSLR(Module):
    """
    data: [seq_len, batch_size, 3], 3 corresponds to [intercept, coef, step_size]

    """
    def __init__(self, in_size, num_hidden, hidden_dim, use_attention=False, device=None):
        super(RNNiSLR, self).__init__()
        if device is None and tc.cuda.is_available():
            self.device = "cuda: 2"
        elif device is None and tc.backends.mps.is_available():
            self.device = "mps"
        elif device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.dropout_p = [0.5, 0.3] if self.training else [0,0]
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim

        self.fc_in = Linear(3, in_size, device=self.device)
        self.lstm_enc = LSTM(input_size=in_size, hidden_size=hidden_dim, num_layers=num_hidden, dropout=self.dropout_p[1], device=self.device)

        self.use_attention = use_attention
        if self.use_attention:
            self.att_q = Linear(in_size, hidden_dim, device=self.device)
            self.att_k = Linear(in_size, hidden_dim, device=self.device)
            self.att_v = Linear(hidden_dim, hidden_dim, device=self.device)
            self.att = MultiheadAttention(hidden_dim, num_hidden, device=self.device)

        self.lstm_dec = LSTM(input_size=hidden_dim, hidden_size=in_size, num_layers=num_hidden, dropout=self.dropout_p[1], device=self.device)
        self.fc_out = Linear(in_size, 3, device=self.device)

    def initialize_lstm(self, batch_size, init_hidden=True):
        if init_hidden:
            h0, c0 = tc.zeros([self.num_hidden, batch_size, self.hidden_dim], device=self.device), tc.zeros(
                [self.num_hidden, batch_size, self.hidden_dim], device=self.device)
            return (h0, c0)
        else:
            return None

    def forward(self, x, hidden=None):
        # x: [seq_len, batch_size, 3]
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        batch_size = x.shape[1]
        if any([hidden is None]):
            hidden = self.initialize_lstm(batch_size=batch_size)

        x_in = Sequential(self.fc_in, Dropout(0.5))(x) if self.training else self.fc_in(x) # x_in: [seq_len, batch_size, in_size]
        state, (hn, cn) = self.lstm_enc(x_in, hidden) # state: [seq_len, batch_size, hidden_dim]

        if self.use_attention:
            x_q = self.att_q(x_in)
            x_k = self.att_k(x_in)
            z_v = self.att_v(state)
            state,_ = self.att(x_q, x_k, z_v, need_weights=False) # after attention, state: [seq_len, batch_size, hidden_dim]

        x_dec , _ = self.lstm_dec(state, (hn, cn)) # x_dec: [seq_len, batch_size, in_size]
        x_dec = x_dec.swapdims(1,0).mean(1) # x_dec: [batch_size, in_size]
        out = Sequential(self.fc_out, Dropout(0.5))(x_dec) if self.training else self.fc_out(x_dec) # out: [batch_size, 3]

        return out








# inp = tc.randn([5, 10, 3])
#
# lstm_enc = LSTM(input_size=3, hidden_size=3, num_layers=3)
# z, (hn,cn) = lstm_enc(inp, (tc.ones([3, 10, 3]), tc.ones([3,10,3])))
# z.shape
#
# att = MultiheadAttention(6,3)
# att_q = Linear(3, 6)
# att_k = Linear(3, 6)
# att_v = Linear(3, 6)
#
# # emb = Embedding(10,4)
# # emb(tc.ones(3,7, dtype=tc.long)).shape
# inq = att_q(z)
# ink = att_k(z)
# inv = att_v(z)
# at, _ = att(inq, ink, inv, need_weights=False)
# at.shape
#
# lstm_dec = LSTM(input_size=6, hidden_size=3, num_layers=3)
# x, (hnn,cnn) = lstm_dec(at, (hn, cn))
# x.shape
#
#
# lstm_dec.train(mode=False)
# lstm_dec.training