import pickle
import numpy as np
import argparse

from sklearn.ensemble import RandomForestClassifier

from factor_selection import *
import scipy.stats as ss

from utils import *
from cfgs.config import cfg

cycle = 12
sel_factor_num = 10

data_file = "data/hs300_2010_2017_financial_1m.pkl"
factor_list = ["adratio", "arturndays", "arturnover", "bips", "business_income", "bvps", "cashflowratio", "cashratio", "cf_liabilities", "cf_nm", "cf_sales", "currentasset_days", "currentasset_turnover", "currentratio", "epcf", "eps", "epsg", "gross_profit_rate", "icratio", "inventory_turnover", "mbrg", "nav", "net_profit_ratio", "net_profits", "nprg", "quickratio", "rateofreturn", "roe", "seg", "sheqratio", "BVY", "CF2TA", "SY", "EBT2TA", "EBITDA2TA", "EBITDA", "EBIT", "general_equity", "flow_equity", "EBITDA2TA", "pe"]

index_code = "sh000300"

f = open(data_file, 'rb')
data_seq = pickle.load(f)

period_num = len(data_seq)

corr_rate_list = []

for time_step in range(cycle, period_num - 1):
    cur_data_seq = data_seq[time_step-cycle:time_step]
    factor_ic_list = []
    t_stat_list = []
    for factor_code in factor_list:
        ic_result = calculate_ic(cur_data_seq, factor_code)
        # only use the ic with lag 1
        ic_result = ic_result[0]
        t_stat = ss.ttest_1samp(ic_result, 0)

        factor_ic_list.append(np.mean(ic_result))
        t_stat_list.append(t_stat.statistic)

    # sort factor by rank IC and t_stat, and choose several factors
    ic_sort = np.argsort(factor_ic_list)[::-1]
    sel_factor_idx = ic_sort[:sel_factor_num]

    # use these factors to construct model to predict return
    # 当前时间点是temp_step-1，训练数据从time_step-cycle一直到time_step-1，即time_step-cycle:time_step，共计cycle个时间点
    # 每一个样本都是由上一个时间点的factor，以及上一个时间点的close和下一个时间点的close计算出的return构成的
    # 因此训练样本分布在cycle-1个时间点上，而预测时，是用当前step的因子，去预测下一个step和当前step比较得到的return
    # 若总时间是period（从0到period-1），那么进行预测以及准确率评估应该是从第cycle-1个time step一直到第period-2个time_step
    # 即range(cycle, period_num - 1)

    # collect training samples and test samples
    # cur_data_seq should include one more data point, which is used to generate test data
    cur_data_seq = data_seq[time_step-cycle:time_step + 1]
    train_samples = []
    train_return_vals = []
    for t in range(cycle):

        cur_samples = []
        cur_return_vals = []

        cur_data = cur_data_seq[t]
        next_data = cur_data_seq[t + 1]
        index_return = -1
        for stock_idx, stock_data in cur_data.iterrows():
            if stock_data['code'] == index_code:
                # save the index return
                next_stock_data = next_data.loc[next_data['code'] == index_code]
                if next_stock_data.shape[0] != 1:
                    continue
                close = stock_data['close']
                next_close = next_stock_data.iloc[0]['close']
                index_return = next_close / close - 1

            input_factors = []
            has_nan = False
            for factor_idx in sel_factor_idx:
                factor_name = factor_list[factor_idx]
                factor_val = stock_data[factor_name]
                if np.isnan(factor_val) or np.isinf(factor_val):
                    has_nan = True
                    break
                input_factors.append(factor_val)
            if has_nan:
                continue
            next_stock_data = next_data.loc[next_data['code'] == stock_data['code']]

            if next_stock_data.shape[0] != 1:
                continue
            close = stock_data['close']
            next_close = next_stock_data.iloc[0]['close']

            return_val = next_close / close - 1

            cur_samples.append(np.asarray(input_factors))
            cur_return_vals.append(return_val)

        cur_samples = np.asarray(cur_samples)
        cur_return_vals = np.asarray(cur_return_vals)

        cur_samples = np.argsort(cur_samples, axis=0)
        # cur_return_vals = np.argsort(cur_return_vals)
        cur_return_vals = (cur_return_vals > index_return).astype(np.int)

        if t == cycle - 1:
            test_samples = cur_samples
            test_return_vals = cur_return_vals
        else:
            train_samples.append(cur_samples)
            train_return_vals.append(cur_return_vals)

    train_samples = np.vstack(train_samples)
    train_return_vals = np.hstack(train_return_vals)

    # training and test data for one time step are collected

    clf = RandomForestClassifier(n_estimators=200)
    predict_return_vals = clf.predict(test_samples)
    corr_num = np.sum((predict_return_vals == test_return_vals).astype(np.int))
    corr_rate = corr_num / test_return_vals.shape[0]

    print(corr_rate)
    corr_rate_list.append(corr_rate)

import pdb
pdb.set_trace()

