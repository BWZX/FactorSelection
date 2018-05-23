import pickle
import numpy as np
import argparse

from factor_selection import *
import scipy.stats as ss

from utils import *
from cfgs.config import cfg

cycle = 12
sel_factor_num = 10

financial_data_file = "data/hs300_2010_2017_financial_1m.pkl"
financial_factor_list = ["adratio", "arturndays", "arturnover", "bips", "business_income", "bvps", "cashflowratio", "cashratio", "cf_liabilities", "cf_nm", "cf_sales", "currentasset_days", "currentasset_turnover", "currentratio", "epcf", "eps", "epsg", "gross_profit_rate", "icratio", "inventory_turnover", "mbrg", "nav", "net_profit_ratio", "net_profits", "nprg", "quickratio", "rateofreturn", "roe", "seg", "sheqratio", "BVY", "CF2TA", "SY", "EBT2TA", "EBITDA2TA", "EBITDA", "EBIT", "general_equity", "flow_equity", "EBITDA2TA", "pe"]

technical_data_file = "data/hs300_2010_2017_tech_1_1m.pkl"
technical_factor_list = []

factor_list = financial_factor_list + technical_factor_list

financial_f = open(financial_data_file, 'rb')
financial_data_seq_list = pickle.load(financial_f)
technical_f = open(technical_data_file, 'rb')
technical_data_seq_list = pickle.load(technical_f)


period_num = len(data_seq_list)

for time_step in range(cycle, period_num - 1):
    cur_financial_data_seq = financial_data_seq_list[time_step-cycle:time_step]
    cur_technical_data_seq = technical_data_seq_list[time_step-cycle:time_step]
    factor_ic_list = []
    t_stat_list = []
    for factor_code in factor_list:
        if factor_code in financial_factor_list:
            cur_data_seq = cur_financial_data_seq
        else:
            cur_data_seq = cur_technical_data_seq
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

    # collect training samples
    for t in range(cycle-1):
        financial_data = cur_financial_data_seq_list[t]
        techncial_data = cur_technical_data_seq_list[t]
        next_data = cur_financial_data_seq_list[t + 1]
        code_list = financial_data['code'].tolist()
        close_list = financial_data['close'].tolist()
        next_close_list = next_data['close'].tolist()
