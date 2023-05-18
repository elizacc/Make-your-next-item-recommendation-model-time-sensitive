# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" TiSASRec
Reference:
    "Time Interval Aware Self-Attention for Sequential Recommendation"
    Jiacheng Li et al., WSDM'2020.
CMD example:
    python main.py --model_name TiSASRec --emb_size 64 --num_layers 2 --num_heads 2 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset Grocery_and_Gourmet_Food --version 4 --prediction_time hour,day --train 1 --time_emb_size 1
"""

import torch
import torch.nn as nn
import numpy as np

from models.sequential.SASRec import SASRec


class TiSASRec(SASRec):
    extra_log_args = ['emb_size', 'num_layers', 'num_heads', 'time_max', 'time_emb_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--time_max', type=int, default=512, help='Max time intervals.')
        # 1 проблема с размерностями решается на уровне embedding_dim, 2 – на уровне transformer_block, 8 – добавляется time embedder
        parser.add_argument('--version', type=int, default=2, help='Model version: from 1 to 8')
        parser.add_argument('--n_dft', type=int, default=64, help='The point of DFT.')
        parser.add_argument('--time_emb_size', type=int, default=64, help='The point of DFT.')
        return SASRec.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.max_time = args.time_max
        self.time_emb_size = args.time_emb_size
        self.prediction_time = args.prediction_time
        self.version = 0 if self.prediction_time == 'no' else args.version
        self.freq_dim = args.n_dft // 2 + 1

        super().__init__(args, corpus)
        self.user_min_interval = dict()
        setattr(corpus, 'user_min_interval', dict())
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs = user_df['time'].values
            interval_matrix = np.abs(time_seqs[:, None] - time_seqs[None, :])
            min_interval = np.min(interval_matrix + (interval_matrix <= 0) * 0xFFFF)
            self.user_min_interval[u] = min_interval
            corpus.user_min_interval[u] = min_interval

    def _define_params(self):

        if self.end_item > 0:
            self.item_num += 1
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        if self.prediction_time != 'no':
            if self.version == 1:
                self.it_embeddings = nn.Embedding(self.item_num, int(self.emb_size / 2))
                self.h_shift_embeddings = nn.Embedding(24 + 1, int(self.emb_size / 2))
                self.d_shift_embeddings = nn.Embedding(7 + 1, int(self.emb_size / 2))
                self.m_shift_embeddings = nn.Embedding(12 + 1, int(self.emb_size / 2))
            else:
                self.h_shift_embeddings = nn.Embedding(24 + 1, self.time_emb_size)
                self.d_shift_embeddings = nn.Embedding(7 + 1, self.time_emb_size)
                self.m_shift_embeddings = nn.Embedding(12 + 1, self.time_emb_size)

        if self.version in [4, 5]:
            self.size = self.emb_size + self.time_emb_size
            self.p_k_embeddings = nn.Embedding(self.max_his + 1, self.size)
            self.p_v_embeddings = nn.Embedding(self.max_his + 1, self.size)
            self.t_k_embeddings = nn.Embedding(self.max_time + 1, self.size)
            self.t_v_embeddings = nn.Embedding(self.max_time + 1, self.size)
        else:
            self.p_k_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
            self.p_v_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
            self.t_k_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)
            self.t_v_embeddings = nn.Embedding(self.max_time + 1, self.emb_size)

        if self.version in [4, 5]:
            self.transformer_block = nn.ModuleList([
                TimeIntervalTransformerLayer(d_model=self.size, d_ff=self.emb_size, n_heads=self.num_heads,
                                             dropout=self.dropout, kq_same=False, version=self.version)
                for _ in range(self.num_layers)
            ])
        else:
            self.transformer_block = nn.ModuleList([
                TimeIntervalTransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                             dropout=self.dropout, kq_same=False, version=self.version)
                for _ in range(self.num_layers)
            ])

        self.time_embeddings = TimeEmbedder(hidden_size=self.emb_size, n_freq=self.freq_dim, device=self.device)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        i_history = feed_dict['history_items']  # [batch_size, history_max]
        t_history = feed_dict['history_times']  # [batch_size, history_max]
        user_min_t = feed_dict['user_min_intervals']  # [batch_size]
        lengths = feed_dict['lengths']  # [batch_size]
        history_hours = feed_dict['history_hours']
        history_days = feed_dict['history_days']
        history_months = feed_dict['history_months']
        history_hours_shift = feed_dict['history_hours_shift']
        history_days_shift = feed_dict['history_days_shift']
        history_months_shift = feed_dict['history_months_shift']
        batch_size, seq_len = i_history.shape

        valid_his = (i_history > 0).long()

        if self.version in [0, 8]:
            his_vectors = self.i_embeddings(i_history)

        else:
            if self.prediction_time == 'hour':
                time_embeddings = self.h_shift_embeddings(history_hours)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift)
            elif self.prediction_time == 'day':
                time_embeddings = self.d_shift_embeddings(history_days)
                time_shift_embeddings = self.d_shift_embeddings(history_days_shift)
            elif self.prediction_time == 'month':
                time_embeddings = self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.m_shift_embeddings(history_months_shift)
            elif self.prediction_time == 'hour,day':
                time_embeddings = self.h_shift_embeddings(history_hours) + self.d_shift_embeddings(history_days)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.d_shift_embeddings(
                    history_days_shift)
            elif self.prediction_time == 'hour,month':
                time_embeddings = self.h_shift_embeddings(history_hours) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.m_shift_embeddings(
                    history_months_shift)
            elif self.prediction_time == 'day,month':
                time_embeddings = self.d_shift_embeddings(history_days) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.d_shift_embeddings(history_days_shift) + self.m_shift_embeddings(
                    history_months_shift)
            elif self.prediction_time == 'hour,day,month':
                time_embeddings = self.h_shift_embeddings(history_hours) + self.d_shift_embeddings(
                    history_days) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.d_shift_embeddings(
                    history_days_shift) + self.m_shift_embeddings(history_months_shift)

            # if self.prediction_time == 'hour':
            #     history_times = history_hours
            #     history_times_shift = history_hours_shift
            # elif self.prediction_time == 'day':
            #     history_times = history_days
            #     history_times_shift = history_days_shift
            # elif self.prediction_time == 'month':
            #     history_times = history_months
            #     history_times_shift = history_months_shift

            if self.version == 1:  # last dim = emb_size
                # his_vectors = torch.cat((self.it_embeddings(i_history), self.t_embeddings(history_times_shift)), dim=2)
                his_vectors = torch.cat((self.it_embeddings(i_history), time_shift_embeddings), dim=2)

            elif self.version == 2:  # last dim = emb_size * 2
                # his_vectors = torch.cat((self.i_embeddings(i_history), self.t_embeddings(history_times_shift)), dim=2)
                his_vectors = torch.cat((self.i_embeddings(i_history), time_shift_embeddings), dim=2)

            elif self.version == 3:
                his_vectors = torch.cat((self.i_embeddings(i_history), time_embeddings), dim=2)

            elif self.version == 4:
                his_vectors = torch.cat((self.i_embeddings(i_history), time_shift_embeddings), dim=2)

            elif self.version == 5:
                his_vectors = torch.cat((self.i_embeddings(i_history), time_embeddings), dim=2)

            elif self.version == 6:
                his_vectors = self.i_embeddings(i_history) + time_shift_embeddings

            elif self.version == 7:
                his_vectors = self.i_embeddings(i_history) + time_embeddings

        # Position embedding
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_k = self.p_k_embeddings(position)
        pos_v = self.p_v_embeddings(position)

        # Time Embedder
        if self.version == 8:
            valid_mask = valid_his.view(batch_size, seq_len, 1)
            time_vectors = self.time_embeddings(t_history, valid_mask)
            pos_k += time_vectors
            pos_v += time_vectors

        # Interval embedding
        interval_matrix = (t_history[:, :, None] - t_history[:, None, :]).abs()
        interval_matrix = (interval_matrix / user_min_t.view(-1, 1, 1)).long().clamp(0, self.max_time)
        inter_k = self.t_k_embeddings(interval_matrix)
        inter_v = self.t_v_embeddings(interval_matrix)

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, pos_k, pos_v, inter_k, inter_v, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids)
        if self.version in [4, 5]:
            if self.prediction_time == 'hour':
                d = torch.tensor(np.tile(feed_dict['hours'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(
                    self.device)
                d = self.h_shift_embeddings(d)
                i_vectors = torch.cat((i_vectors, d), 2)
            elif self.prediction_time == 'day':
                d = torch.tensor(np.tile(feed_dict['days'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(
                    self.device)
                d = self.d_shift_embeddings(d)
                i_vectors = torch.cat((i_vectors, d), 2)
            elif self.prediction_time == 'hour,day':
                h = torch.tensor(np.tile(feed_dict['hours'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(
                    self.device)
                h = self.h_shift_embeddings(h)
                d = torch.tensor(np.tile(feed_dict['days'].cpu(), (i_vectors.shape[1], 1)).transpose()).to(
                    self.device)
                d = self.d_shift_embeddings(d)
                i_vectors = torch.cat((i_vectors, h+d), 2)

        prediction = (his_vector[:, None, :] * i_vectors).sum(-1)
        return {'prediction': prediction.view(batch_size, -1)}

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples (a little different now)
        "Recurrent neural networks with top-k gains for session-based recommendations"
        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        if self.prediction_time == 'hour':
            weights_loss = torch.sum(self.h_shift_embeddings.weight**2)
        elif self.prediction_time == 'day':
            weights_loss = torch.sum(self.d_shift_embeddings.weight ** 2)
        elif self.prediction_time == 'hour,day':
            weights_loss = torch.sum(self.h_shift_embeddings.weight**2) + torch.sum(self.d_shift_embeddings.weight ** 2)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean() + 0 * weights_loss
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, we use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss

    class Dataset(SASRec.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            user_id = self.data['user_id'][index]
            min_interval = self.corpus.user_min_interval[user_id]
            feed_dict['user_min_intervals'] = min_interval
            return feed_dict


class TimeIntervalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True, version=0):
        super().__init__()
        """
        It also needs position and interaction (time interval) key/value input.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same
        self.version = version

        if self.version in [2, 3]:
            self.v_linear = nn.Linear(d_model * 2, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model * 2, d_model, bias=bias)
            if not kq_same:
                self.q_linear = nn.Linear(d_model * 2, d_model, bias=bias)
        else:
            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if not kq_same:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, pos_k, pos_v, inter_k, inter_v, mask):
        bs, seq_len = k.size(0), k.size(1)

        # perform linear operation and split into h heads
        k = (self.k_linear(k) + pos_k).view(bs, seq_len, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, seq_len, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, seq_len, self.h, self.d_k)
        v = (self.v_linear(v) + pos_v).view(bs, seq_len, self.h, self.d_k)

        # transpose to get dimensions bs * h * -1 * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # interaction (time interval) embeddings
        inter_k = inter_k.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_v = inter_v.view(bs, seq_len, seq_len, self.h, self.d_k)
        inter_k = inter_k.transpose(2, 3).transpose(1, 2)
        inter_v = inter_v.transpose(2, 3).transpose(1, 2)  # bs, head, seq_len, seq_len, d_k

        # calculate attention using function we will define next
        output = self.scaled_dot_product_attention(q, k, v, inter_k, inter_v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        output = output.transpose(1, 2).reshape(bs, -1, self.d_model)
        return output

    @staticmethod
    def scaled_dot_product_attention(q, k, v, inter_k, inter_v, d_k, mask):
        """
        Involve pair interaction embeddings when calculating attention scores and output
        """
        scores = torch.matmul(q, k.transpose(-2, -1))  # bs, head, q_len, k_len
        scores += (q[:, :, :, None, :] * inter_k).sum(-1)
        scores = scores / d_k ** 0.5
        scores.masked_fill_(mask == 0, -np.inf)
        scores = (scores - scores.max()).softmax(dim=-1)
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
        output += (scores[:, :, :, :, None] * inter_v).sum(-2)
        return output


class TimeIntervalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same=False, version=0):
        super().__init__()
        self.version = version
        self.masked_attn_head = TimeIntervalMultiHeadAttention(d_model, n_heads, kq_same=kq_same, version=version)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        if version in [2, 3]:
            self.linear = nn.Linear(d_model * 2, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, pos_k, pos_v, inter_k, inter_v, mask):
        context = self.masked_attn_head(seq, seq, seq, pos_k, pos_v, inter_k, inter_v, mask)  # last dim = d_model
        if self.version in [2, 3]:
            context = self.layer_norm1(self.dropout1(context) + self.linear(seq))  # seq last dim = d_model * 2
        else:
            context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output


# класс TimeEmbedder аналогично RelationalDynamicAggregation в KDA
class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, n_freq, device):
        super().__init__()
        self.freq_real = nn.Embedding(hidden_size, n_freq)
        self.freq_imag = nn.Embedding(hidden_size, n_freq)
        freq = np.linspace(0, 1, n_freq) / 2.
        self.freqs = torch.from_numpy(np.concatenate((freq, -freq))).to(device).float()
        self.range = torch.from_numpy(np.arange(hidden_size)).to(device)

    def idft_decay(self, t):
        real, imag = self.freq_real(self.range), self.freq_imag(self.range)
        # create conjugate symmetric to ensure real number output
        x_real = torch.cat([real, real], dim=-1)
        x_imag = torch.cat([imag, -imag], dim=-1)
        w = 2. * np.pi * self.freqs * t.unsqueeze(-1)  # B * H * n_freq
        real_part = w.cos()[:, :, None, :] * x_real[None, None, :, :]  # B * H * R * n_freq
        imag_part = w.sin()[:, :, None, :] * x_imag[None, None, :, :]
        decay = (real_part - imag_part).mean(dim=-1) / 2.  # B * H * R
        return decay.float()

    def forward(self, t, valid_mask):
        decay = self.idft_decay(t).clamp(0, 1).masked_fill(valid_mask == 0, 0.)
        return decay