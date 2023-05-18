# -*- coding: UTF-8 -*-

import os
import subprocess
from sys import stdout
import pandas as pd
import argparse
import re
import traceback
import numpy as np
from typing import List


# Repeat experiments and save results to csv
# Example: python exp.py --in_f run.sh --out_f exp.csv --n 5


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--log_dir', nargs='?', default='../log/',
                        help='Log save dir.')
    parser.add_argument('--cmd_dir', nargs='?', default='./',
                        help='Command dir.')
    parser.add_argument('--in_f', nargs='?', default='run.sh',
                        help='Input commands.')
    parser.add_argument('--out_f', nargs='?', default='exp.csv',
                        help='Output csv.')
    parser.add_argument('--base_seed', type=int, default=0,
                        help='Random seed at the beginning.')
    parser.add_argument('--n', type=int, default=5,
                        help='Repeat times of each command.')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip number.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def find_info(result: List[str]) -> dict:
    info = dict()
    prefix = ''
    for line in result:
        # if line.startswith(prefix + 'Best Iter(dev)'):
        #     line = line.replace(' ', '')
        #     p = re.compile('BestIter\(dev\)=(\d*)')
        #     info['Best Iter'] = p.search(line).group(1)
        #     p = re.compile('\[([\d\.]+)s\]')
        #     info['Time'] = p.search(line).group(1)
        # elif line.startswith(prefix + 'Test After Training:'):
        #     p = re.compile('\(([\w@:\.\d,]+)\)')
        #     info['Test'] = p.search(line).group(1)
        if line.startswith(prefix + 'Early stop at'):
            p = re.compile('\d+')
            info['Epochs'] = p.search(line).group(0)
        if line.startswith(prefix + 'Test After Training:'):
            p = re.compile('\(([\w@:\.\d,]+)\)')
            metrics = p.search(line).group(1)
            for m in metrics.split(','):
                metric_name, metric_value = m.split(':')
                info[metric_name] = float(metric_value)

    return info


def main():
    args = parse_args()
    # columns = ['Model', 'Test', 'Best Iter', 'Time', 'Seed', 'Run CMD']
    # columns = ['Model', 'Version', 'Train', 'Prediction hour', 'HR@5', 'HR@10', 'HR@20', 'HR@50', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
    columns = ['Model', 'Version', 'Train', 'Emb size', 'Prediction hour', 'Prediction weekday', 'Prediction month', 'Epochs', 
                'HR@5', 'HR@10', 'HR@20', 'HR@50', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
    skip = args.skip
    df = pd.DataFrame(columns=columns)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    in_f = open(os.path.join(args.cmd_dir, args.in_f), 'r')
    lines = in_f.readlines()

    # Iterate commands
    for cmd in lines:
        cmd = cmd.strip()
        if cmd == '' or cmd.startswith('#') or cmd.startswith('export') or cmd.startswith('for') or cmd == 'do' or cmd == 'done':
            continue
        p = re.compile('--model_name (\w+)')
        model_name = p.search(cmd).group(1)
        p = re.compile('--version (\w+)')
        model_version = p.search(cmd).group(1)
        p = re.compile('--train (\w+)')
        model_train = p.search(cmd).group(1)
        p = re.compile('--emb_size (\w+)')
        emb_size = p.search(cmd).group(1)
        p = re.compile('--prediction_hour (\w+)')
        prediction_hour = p.search(cmd).group(1)
        p = re.compile('--prediction_weekday (\w+)')
        prediction_weekday = p.search(cmd).group(1)
        p = re.compile('--prediction_month (\w+)')
        prediction_month = p.search(cmd).group(1)

        # Repeat experiments
        for i in range(args.base_seed, args.base_seed + args.n):
            try:
                command = cmd
                if command.find(' --random_seed') == -1:
                    command += ' --random_seed ' + str(i)
                if command.find(' --gpu ') == -1:
                    command += ' --gpu ' + args.gpu
                if command.find(' --prediction_hour ') == -1:
                    command += ' --prediction_hour ' + args.prediction_hour
                if command.find(' --prediction_weekday ') == -1:
                    command += ' --prediction_weekday ' + args.prediction_weekday
                if command.find(' --prediction_month ') == -1:
                    command += ' --prediction_month ' + args.prediction_month
                print(command)
                if skip > 0:
                    skip -= 1
                    continue
                # result = subprocess.check_output(command, shell=True)
                result = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
                print(result)
                # result = result.decode('utf-8')
                result = [line.strip() for line in result.split(os.linesep)]
                info = find_info(result)
                # info['Seed'] = str(i)
                # info['Run CMD'] = command
                # if args.n == 1:
                info['Model'] = model_name
                info['Version'] = model_version
                info['Train'] = model_train
                info['Emb size'] = emb_size
                info['Prediction hour'] = prediction_hour
                info['Prediction weekday'] = prediction_weekday
                info['Prediction month'] = prediction_month
                row = [info[c] if c in info else '' for c in columns]
                df.loc[len(df)] = row
                df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)
                # print(df[columns[:5]])
            except Exception as e:
                traceback.print_exc()
                continue

        # Average results
        if args.n > 1:
            info = {'Model': model_name}
            tests = df['Test'].tolist()[-args.n:]
            tests = [[float(m.split(':')[1]) for m in t.split(',')] for t in tests]
            avgs = ['{:<.4f}'.format(np.average([t[mi] for t in tests])) for mi in range(len(tests[0]))]
            info['Test'] = ','.join(avgs)
            row = [info[c] if c in info else '' for c in columns]
            df.loc[len(df)] = row
            # print(df[columns[:5]])
        # for i in range(3):
        #     row = [''] * len(columns)
        #     df.loc[len(df)] = row

        # Save results
        df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)


if __name__ == '__main__':
    main()
