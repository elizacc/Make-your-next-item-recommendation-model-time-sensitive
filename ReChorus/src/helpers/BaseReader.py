# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from typing import NoReturn

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset

        self._read_data()
        self._append_his_info()

    def _read_data(self) -> NoReturn:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        self.all_df = pd.concat([df[['user_id', 'item_id', 'time']] for df in self.data_df.values()])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        # if self.dataset.startswith('recsys_2022_submit'):
        #     self.n_users = self.all_df['user_id'].nunique() + 1
        #     candidate_items = pd.read_csv('../../a.a.sverkunova/my_experiments/dressipi_recsys2022/candidate_items.csv')
        #     self.n_items = len(set(self.all_df['item_id']) | set(candidate_items['item_id'].values)) + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))

    def _append_his_info(self) -> NoReturn:
        """
        Add history info to data_df: position
        ! Need data_df to be sorted by time in ascending order
        """
        logging.info('Appending history info...')
        self.user_his = dict()  # store the already seen sequence of each user
        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.clicked_set = dict()

        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            position = list()

            if df.shape[1] <= 5:
                for uid, iid, t, tsh in zip(df['user_id'], df['item_id'], df['time'], df['time_shift']):
                    if uid not in self.user_his:
                        self.user_his[uid] = list()
                        self.train_clicked_set[uid] = set()
                        self.clicked_set[uid] = set()
                    position.append(len(self.user_his[uid]))
                    self.user_his[uid].append((iid, t, tsh))
                    if key == 'train':
                        self.train_clicked_set[uid].add(iid)
                    self.clicked_set[uid].add(iid)
                df['position'] = position
            
            else:
                for uid, iid, t, y, m, d, h, dw, t0, t1, t2, t3, t4, iid0, iid1, iid2, iid3, iid4, m0, m1, m2, m3, m4,\
                    d0, d1, d2, d3, d4, h0, h1, h2, h3, h4, dw0, dw1, dw2, dw3, dw4 in \
                    zip(df['user_id'], df['item_id'], df['time'], df['year'], df['month'], df['date'], df['hour'], 
                        df['day_of_week'], df['time0'], df['time1'], df['time2'], df['time3'], df['time4'],
                        df['item_id0'], df['item_id1'], df['item_id2'], df['item_id3'], df['item_id4'], 
                        df['month0'], df['month1'], df['month2'], df['month3'], df['month4'], 
                        df['date0'], df['date1'], df['date2'], df['date3'], df['date4'], 
                        df['hour0'], df['hour1'], df['hour2'], df['hour3'], df['hour4'], 
                        df['day_of_week0'], df['day_of_week1'], df['day_of_week2'], df['day_of_week3'], df['day_of_week4']):
                    if uid not in self.user_his:
                        self.user_his[uid] = list()
                        self.train_clicked_set[uid] = set()
                        self.clicked_set[uid] = set()
                    position.append(len(self.user_his[uid]))
                    self.user_his[uid].append((iid, t, y, m, d, h, dw, t0, t1, t2, t3, t4, iid0, iid1, iid2, iid3, iid4, 
                        m0, m1, m2, m3, m4, d0, d1, d2, d3, d4, h0, h1, h2, h3, h4, dw0, dw1, dw2, dw3, dw4))
                    if key == 'train':
                        self.train_clicked_set[uid].add(iid)
                    self.clicked_set[uid].add(iid)
                df['position'] = position


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser = BaseReader.parse_data_args(parser)
    args, extras = parser.parse_known_args()

    args.path = '../../data/'
    corpus = BaseReader(args)

    corpus_path = os.path.join(args.path, args.dataset, 'BaseReader.pkl')
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(corpus, open(corpus_path, 'wb'))
