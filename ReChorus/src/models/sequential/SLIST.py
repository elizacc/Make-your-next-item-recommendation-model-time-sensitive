""" SLIST
Reference:
    "Session-aware Linear Item-Item Models for Session-based Recommendation"
    Minjin Choi et al., WWW'2021.
CMD example:
    python main.py --model_name SLIST --train 0 --fullseq 1 --history_max 0 --dataset 'ml-100k'
"""

import torch
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, vstack, diags
from sklearn.preprocessing import normalize
from tqdm import tqdm

from models.BaseModel import SequentialModel

import warnings
warnings.simplefilter('ignore')


class SLIST(SequentialModel):
    """
    Should run with --train 0, --fullseq 1
    """
    extra_log_args = ['reg', 'alpha', 'session_weight', 'train_weight', 'predict_weight',
                      'direction', 'normalize', 'epsilon', 'user_max_time', 'pos_time']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--reg', type=int, default=10)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--session_weight', type=int, default=-1)
        parser.add_argument('--train_weight', type=int, default=-1)
        parser.add_argument('--predict_weight', type=int, default=-1)
        parser.add_argument('--direction', type=str, default='part')
        parser.add_argument('--normalize', type=str, default='l1')
        parser.add_argument('--epsilon', type=float, default=10.0)
        parser.add_argument('--user_max_time', type=int, default=1,
                            help='Whether to use max time as in the original paper or max user time')
        parser.add_argument('--pos_time', type=int, default=0,
                            help='Whether to use times instead of positions')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.reg = args.reg
        self.normalize = args.normalize
        self.epsilon = args.epsilon
        self.alpha = args.alpha
        self.direction = args.direction
        self.train_weight = float(args.train_weight)
        self.predict_weight = float(args.predict_weight)
        self.session_weight = args.session_weight * 24 * 3600
        self.user_max_time = args.user_max_time
        self.pos_time = args.pos_time  # замена позиций на время
        if self.pos_time > 0:
            self.train_weight = float(args.train_weight) * 24 * 3600  # * 1000000

        self.max_time = corpus.all_df['time'].max()
        self.data = pd.concat([corpus.data_df['train'], corpus.data_df['dev']])

        super().__init__(args, corpus)

    def _define_params(self):
        # ||X - XB||
        input1, target1, row_weight1 = self._make_train_matrix(weight_by='SLIS')
        # ||Y - ZB||
        input2, target2, row_weight2 = self._make_train_matrix(weight_by='SLIT')
        # alpha * ||X - XB|| + (1-alpha) * ||Y - ZB||
        input1.data = np.sqrt(self.alpha) * input1.data
        target1.data = np.sqrt(self.alpha) * target1.data
        input2.data = np.sqrt(1 - self.alpha) * input2.data
        target2.data = np.sqrt(1 - self.alpha) * target2.data

        input_matrix = vstack([input1, input2])
        target_matrix = vstack([target1, target2])
        w2 = row_weight1 + row_weight2  # list

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        W2 = diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        print(f"G is made. Sparsity: {(1 - np.count_nonzero(G) / (self.item_num ** 2)) * 100}%")

        P = np.linalg.inv(G + np.identity(self.item_num, dtype=np.float32) * self.reg)
        print("P is made")
        del G

        if self.alpha == 1:
            C = -P @ (input_matrix.transpose().dot(W2).dot(input_matrix - target_matrix).toarray())

            mu = np.zeros(self.item_num)
            mu += self.reg
            mu_nonzero_idx = np.where(1 - np.diag(P) * self.reg + np.diag(C) >= self.epsilon)
            mu[mu_nonzero_idx] = (np.diag(1 - self.epsilon + C) / np.diag(P))[mu_nonzero_idx]

            # B = I - Pλ + C
            self.enc_w = np.identity(self.item_num, dtype=np.float32) - P @ np.diag(mu) + C
            print("Weight matrix is made")
        else:
            self.enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()

    def _make_train_matrix(self, weight_by='SLIT'):
        data = self.data.sort_values('item_id')

        input_row = []
        target_row = []
        input_col = []
        target_col = []
        input_data = []
        target_data = []

        w2 = []
        sessionlengthmap = data['user_id'].value_counts(sort=False)
        rowid = -1

        if weight_by == 'SLIT':
            for sid, session in tqdm(data.groupby('user_id'), desc=weight_by):
                sessionitems = session.sort_values('time')['item_id'].tolist()  # sorted by time
                sessiontimes = session.sort_values('time')['time'].tolist()
                slen = len(sessionitems)
                if slen <= 1:
                    continue
                stime = session['time'].max()
                if self.user_max_time > 0:
                    maxtime = session['time_shift'].max()
                else:
                    maxtime = self.max_time  # проверка макс времени как в оригинале
                w2 += [stime - maxtime] * (slen - 1)

                if self.pos_time == 0:
                    for t in range(slen - 1):
                        rowid += 1

                        # input matrix
                        if self.direction == 'part':
                            input_row += [rowid] * (t + 1)
                            input_col += sessionitems[:t + 1]
                            for s in range(t + 1):
                                input_data.append(-abs(t - s))
                            target_row += [rowid] * (slen - (t + 1))
                            target_col += sessionitems[t + 1:]
                            for s in range(t + 1, slen):
                                target_data.append(-abs((t + 1) - s))

                        elif self.direction == 'all':
                            input_row += [rowid] * slen
                            input_col += sessionitems
                            for s in range(slen):
                                input_data.append(-abs(t - s))
                            target_row += [rowid] * slen
                            target_col += sessionitems
                            for s in range(slen):
                                target_data.append(-abs((t + 1) - s))

                        elif self.direction == 'sr':
                            input_row += [rowid]
                            input_col += [sessionitems[t]]
                            input_data.append(0)
                            target_row += [rowid] * (slen - (t + 1))
                            target_col += sessionitems[t + 1:]
                            for s in range(t + 1, slen):
                                target_data.append(-abs((t + 1) - s))

                        else:
                            raise "You have to choose right 'direction'!"

                # замена позиций на время
                else:
                    for t in range(slen - 1):
                        rowid += 1

                        # input matrix
                        if self.direction == 'part':
                            input_row += [rowid] * (t + 1)
                            input_col += sessionitems[:t + 1]
                            for s in range(t + 1):
                                input_data.append(-abs(sessiontimes[t] - sessiontimes[s]))
                            target_row += [rowid] * (slen - (t + 1))
                            target_col += sessionitems[t + 1:]
                            for s in range(t + 1, slen):
                                target_data.append(-abs((sessiontimes[t + 1]) - sessiontimes[s]))

                        elif self.direction == 'all':
                            input_row += [rowid] * slen
                            input_col += sessionitems
                            for s in range(slen):
                                input_data.append(-abs(sessiontimes[t] - sessiontimes[s]))
                            target_row += [rowid] * slen
                            target_col += sessionitems
                            for s in range(slen):
                                target_data.append(-abs((sessiontimes[t + 1]) - sessiontimes[s]))

                        elif self.direction == 'sr':
                            input_row += [rowid]
                            input_col += [sessionitems[t]]
                            input_data.append(0)
                            target_row += [rowid] * (slen - (t + 1))
                            target_col += sessionitems[t + 1:]
                            for s in range(t + 1, slen):
                                target_data.append(-abs((sessiontimes[t + 1]) - sessiontimes[s]))

                        else:
                            raise "You have to choose right 'direction'!"

            input_data = list(np.exp(np.array(input_data) / self.train_weight))
            target_data = list(np.exp(np.array(target_data) / self.train_weight))

        elif weight_by == 'SLIS':
            for sid, session in tqdm(data.groupby(['user_id']), desc=weight_by):
                rowid += 1
                slen = sessionlengthmap[sid]
                sessionitems = session['item_id'].tolist()
                stime = session['time'].max()
                if self.user_max_time > 0:
                    maxtime = session['time_shift'].max()
                else:
                    maxtime = self.max_time  # проверка макс времени как в оригинале
                w2.append(stime - maxtime)
                input_row += [rowid] * slen
                input_col += sessionitems

            target_row = input_row
            target_col = input_col
            input_data = np.ones_like(input_row)
            target_data = np.ones_like(target_row)

        else:
            raise "You have to choose right 'weight_by'!"

        # Use train_weight or not
        input_data = input_data if self.train_weight > 0 else list(np.ones_like(input_data))
        target_data = target_data if self.train_weight > 0 else list(np.ones_like(target_data))

        # Use session_weight or not
        w2 = list(np.exp(np.array(w2) / self.session_weight))
        w2 = w2 if self.session_weight > 0 else list(np.ones_like(w2))

        # Make sparse_matrix
        input_matrix = csr_matrix((input_data, (input_row, input_col)),
                                  shape=(max(input_row) + 1, self.item_num), dtype=np.float32)
        target_matrix = csr_matrix((target_data, (target_row, target_col)),
                                   shape=input_matrix.shape, dtype=np.float32)
        print(
            f"[{weight_by}] sparse matrix {input_matrix.shape} is made. Sparsity: "
            f"{(1 - input_matrix.count_nonzero() / (self.item_num * input_matrix.shape[0])) * 100}%"
        )

        if weight_by == 'SLIT':
            pass
        elif weight_by == 'SLIS':
            # Value of repeated items --> 1
            input_matrix.data = np.ones_like(input_matrix.data)
            target_matrix.data = np.ones_like(target_matrix.data)

        # Normalization
        if self.normalize == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif self.normalize == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass

        return input_matrix, target_matrix, w2

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        i_history = feed_dict['history_items']  # [batch_size, history_max]

        prediction = []
        # items_to_predict = np.arange(1, self.item_num)

        for b in range(feed_dict['batch_size']):
            items_to_predict = i_ids[b]
            session_items = i_history[b]
            W_test = np.ones_like(session_items, dtype=np.float32)
            W_test = self.enc_w[session_items[-1], session_items]
            for i in range(len(W_test)):
                W_test[i] = np.exp(-abs(i + 1 - len(W_test)) / self.predict_weight)

            W_test = W_test if self.predict_weight > 0 else np.ones_like(W_test)
            W_test = W_test.reshape(-1, 1)

            # [session_items, num_items]
            preds = self.enc_w[session_items] * W_test
            # [num_items]
            preds = np.sum(preds, axis=0)
            preds = preds[items_to_predict]
            prediction.append(preds / np.max(preds))

        prediction = torch.tensor(prediction).to(self.device)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
