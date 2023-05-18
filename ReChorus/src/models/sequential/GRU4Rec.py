# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" GRU4Rec
Reference:
    "Session-based Recommendations with Recurrent Neural Networks"
    Hidasi et al., ICLR'2016.
CMD example:
    python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 128 --lr 1e-3 --l2 1e-4 --history_max 20 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn

from models.BaseModel import SequentialModel


class GRU4Rec(SequentialModel):
    extra_log_args = ['emb_size', 'hidden_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--version', type=int, default=2) 
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.prediction_time = args.prediction_time
        self.version = 0 if self.prediction_time == 'no' else args.version
        super().__init__(args, corpus)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.h_shift_embeddings = nn.Embedding(24 + 1, self.emb_size)
        self.d_shift_embeddings = nn.Embedding(7 + 1, self.emb_size)
        self.m_shift_embeddings = nn.Embedding(12 + 1, self.emb_size)
        if self.version in [2, 3]:
            self.rnn = nn.GRU(input_size=self.emb_size * 2, hidden_size=self.hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        # self.pred_embeddings = nn.Embedding(self.item_num, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.emb_size)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        history_hours = feed_dict['history_hours']
        history_days = feed_dict['history_days']
        history_months = feed_dict['history_months']
        history_hours_shift = feed_dict['history_hours_shift']
        history_days_shift = feed_dict['history_days_shift']
        history_months_shift = feed_dict['history_months_shift']

        if self.prediction_time == 'no':  # self.version == 0
            his_vectors = self.i_embeddings(history)

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
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.d_shift_embeddings(history_days_shift)
            elif self.prediction_time == 'hour,month':
                time_embeddings = self.h_shift_embeddings(history_hours) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.m_shift_embeddings(history_months_shift)
            elif self.prediction_time == 'day,month':
                time_embeddings = self.d_shift_embeddings(history_days) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.d_shift_embeddings(history_days_shift) + self.m_shift_embeddings(history_months_shift)
            elif self.prediction_time == 'hour,day,month':
                time_embeddings = self.h_shift_embeddings(history_hours) + self.d_shift_embeddings(history_days) + self.m_shift_embeddings(history_months)
                time_shift_embeddings = self.h_shift_embeddings(history_hours_shift) + self.d_shift_embeddings(history_days_shift) + self.m_shift_embeddings(history_months_shift)

            if self.version == 2:
                his_vectors = torch.cat((self.i_embeddings(history), time_shift_embeddings), dim=2)
            if self.version == 3:
                his_vectors = torch.cat((self.i_embeddings(history), time_embeddings), dim=2)

        # Sort and Pack
        sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)

        # RNN
        output, hidden = self.rnn(history_packed, None)

        # Unsort
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)

        # Predicts
        # pred_vectors = self.pred_embeddings(i_ids)
        pred_vectors = self.i_embeddings(i_ids)
        rnn_vector = self.out(rnn_vector)
        prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
