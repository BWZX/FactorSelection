import pickle
import argparse

from factor_selection import *
import scipy.stats as ss

from utils import *
from cfgs.config import cfg


'''
parser = argparse.ArgumentParser()
parser.add_argument('--factor_code', required=True, help='comma separated factor names')
parser.add_argument('--start_year', type=int, default=2010)
parser.add_argument('--end_year', type=int, default=2017)
parser.add_argument('--period', default='1m')
parser.add_argument('--data_file', default="allstocks_2010_2017_financial_1m.pkl")
args = parser.parse_args()
'''

data_file = "data/hs300_2010_2017_financial_1m.pkl"
cycle = 12
factor_list = ["adratio", "arturndays", "arturnover", "bips", "business_income", "bvps", "cashflowratio", "cashratio", "cf_liabilities", "cf_nm", "cf_sales", "currentasset_days", "currentasset_turnover", "currentratio", "epcf", "eps", "epsg", "gross_profit_rate", "icratio", "inventory_turnover", "mbrg", "nav", "net_profit_ratio", "net_profits", "nprg", "quickratio", "rateofreturn", "roe", "seg", "sheqratio", "BVY", "CF2TA", "SY", "EBT2TA", "EBITDA2TA", "EBITDA", "EBIT", "general_equity", "flow_equity", "EBITDA2TA", "pe"]


f = open(data_file, 'rb')
data_seq_list = pickle.load(f)

for time_step in range(cycle, len(data_seq_list)):
    cur_data_seq = data_seq_list[time_step-cycle:time_step]
    factor_ic_list = []
    t_stat_list = []
    for factor_code in factor_list:
        ic_result = calculate_ic(cur_data_seq, factor_code)
        # only use the ic with lag 1
        ic_result = ic_result[0]
        t_stat = ss.ttest_1samp(ic_result, 0)

        factor_ic_list.append(np.mean(ic_result))
        t_stat_list.append(t_stat.statistic)

    import pdb
    pdb.set_trace()

    # sort factor by rank IC and t_stat, and choose several factors


    # use these factors to construct model to predict return
