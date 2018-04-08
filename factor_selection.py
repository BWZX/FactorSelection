import numpy as np
import os
import pickle
from scipy.stats.stats import pearsonr
import scipy.stats as ss

import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from opdata import opdata

from utils import *
from data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--factor_code', default='eps')
parser.add_argument('--start_year', type=int, default=2010)
parser.add_argument('--end_year', type=int, default=2017)
parser.add_argument('--nr_fractile', type=int, default=5)
parser.add_argument('--equ_wt_benchmark', type=bool, default=True)
args = parser.parse_args()

month_data_ary = load_data('data_hs300_2010_2017_1m.pkl', args.start_year, args.end_year)

output = Output(args.factor_code, args.start_year, args.end_year)

# calculate rank ics for different lags
rank_ic_ary_list = []
for _ in range(12):
    rank_ic_ary_list.append([])

for month_data_idx in range(len(month_data_ary) - 1):

    month_data = month_data_ary[month_data_idx]
    code_list = month_data['code_x'].tolist()
    factor_list = month_data[args.factor_code].tolist()
    close_list = month_data['close'].tolist()

    for lag in range(1, 13):
        if month_data_idx + lag >= len(month_data_ary):
            continue
        future_month_data = month_data_ary[month_data_idx + 1]
        future_code_list = future_month_data['code_x'].tolist()
        future_close_list = future_month_data['close'].tolist()

        # calculate return for each stock in current month list
        return_list = []
        for idx, code in enumerate(code_list):
            current_close = close_list[idx]
            if code not in future_code_list:
                # this stock is not included in the future month data
                return_list.append(None)
                continue
            next_idx = future_code_list.index(code)
            next_close = future_close_list[next_idx]
            return_list.append(next_close / current_close - 1)

        # skip the first code, which should be the index code
        factor_ary = np.asarray(factor_list[1:], dtype=np.float32)
        return_ary = np.asarray(return_list[1:], dtype=np.float32)

        # filter the None value
        factor_filter = factor_ary != None
        return_filter = return_ary != None

        data_filter = factor_filter & return_filter

        factor_ary = factor_ary[data_filter]
        return_ary = return_ary[data_filter]

        factor_rank = get_sort_idx(factor_ary)
        return_rank = get_sort_idx(return_ary)

        rank_ic, _ = pearsonr(return_rank, factor_rank)

        rank_ic_ary_list[lag - 1].append(rank_ic)

# draw rank IC
output.draw_rank_ic(rank_ic_ary_list[0])

# calculate data for decay profile and draw
ave_lag_ic = []
success_rate = []
for rank_ic_ary in rank_ic_ary_list:
    ave_lag_ic.append(np.mean(rank_ic_ary))
    success_rate.append(len([e for e in rank_ic_ary if e > 0]) / len(rank_ic_ary))
output.draw_decay_profile(ave_lag_ic, success_rate)


# calculate fractiles
# for each month, calculate the return for each fractile
fractile_return_list = []
for month_data_idx in range(len(month_data_ary) - 1):

    month_data = month_data_ary[month_data_idx]
    code_list = month_data['code_x'].tolist()
    factor_list = month_data[args.factor_code].tolist()
    close_list = month_data['close'].tolist()

    next_month_data = month_data_ary[month_data_idx + 1]
    next_code_list = next_month_data['code_x'].tolist()
    next_close_list = next_month_data['close'].tolist()

    # calculate return for each stock in current month list
    return_list = []
    for idx, code in enumerate(code_list):
        current_close = close_list[idx]
        if code not in next_code_list:
            # this stock is not included in the future month data
            return_list.append(None)
            continue
        next_idx = next_code_list.index(code)
        next_close = next_close_list[next_idx]
        return_list.append(next_close / current_close - 1)

    if args.equ_wt_benchmark == True:
        stock_return_list = [e for e in return_list[1:] if e is not None]
        return_list[0] = np.mean(stock_return_list)

    # skip the first code, which should be the index code
    factor_ary = np.asarray(factor_list[1:], dtype=np.float32)
    return_ary = np.asarray(return_list[1:], dtype=np.float32)

    # filter the None value
    factor_filter = factor_ary != None

    factor_ary = factor_ary[factor_filter]
    return_ary = return_ary[factor_filter]

    # factor_rank = np.argsort(factor_ary)
    factor_rank = get_sort_idx(factor_ary)

    fractile_loc_list = []
    stock_num = factor_ary.shape[0]
    for idx in range(args.nr_fractile):
        fractile_loc = int(stock_num * (idx + 1) / args.nr_fractile)
        fractile_loc_list.append(fractile_loc)

    return_list_per_fractile = [[] for _ in range(args.nr_fractile)]
    for stock_idx in range(stock_num):
        rank = factor_rank[stock_idx]
        fractile_idx = get_fractile_idx(fractile_loc_list, rank)
        stock_return = return_ary[stock_idx]
        if stock_return != None and np.isnan(stock_return) == False:
            return_list_per_fractile[fractile_idx].append(stock_return)
    ave_return_list = [np.mean(e) for e in return_list_per_fractile]
    ave_return_list.append(return_list[0])

    fractile_return_list.append(ave_return_list)

sim_return = np.ones((args.nr_fractile + 1, len(fractile_return_list) + 1)) * 100

for month_idx, fractile_return in enumerate(fractile_return_list):
    for fractile_idx in range(args.nr_fractile + 1):
        sim_return[fractile_idx, month_idx + 1] = \
            sim_return[fractile_idx, month_idx] * (1 + fractile_return[fractile_idx])

output.draw_fractile(sim_return)

fractile_return_ary = np.transpose(np.asarray(fractile_return_list, dtype=np.float32))

market_metrics = Metrics(fractile_return_ary[-1,:])

fractile_metrics_list = []
for fractile_idx in range(args.nr_fractile):
    fractile_metrics_list.append(Metrics(fractile_return_ary[fractile_idx,:], market_metrics, args.end_year - args.start_year + 1))
fractile_metrics_list.append(market_metrics)


output.show_fractile_table(fractile_metrics_list)
