import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import scipy.stats as ss
import numpy as np

from cfgs.config import cfg

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

def get_sort_idx(items):
    idxes = np.argsort(-items)
    sort_idxes = np.zeros_like(items)
    for idx, ele in enumerate(idxes):
        sort_idxes[ele] = idx
    return sort_idxes

def get_fractile_idx(fractile_loc_list, rank):
    for fractile_idx, loc in enumerate(fractile_loc_list):
        if rank < loc:
            return fractile_idx
    return -1


class Output:
    def __init__(self, factor_code, start_year, end_year, result_dir, period):
        self.factor_code = factor_code
        self.start_year = start_year
        self.end_year = end_year
        self.result_dir = os.path.join(cfg.result_dir, result_dir)
        if os.path.isdir(self.result_dir) == False:
            os.makedirs(self.result_dir)
        self.period = period

    def _filename(self, name):
        return '%s/%s (%d-%d): %s.png' % (self.result_dir, self.factor_code, self.start_year, self.end_year, name)

    def draw_rank_ic(self, rank_ic_ary):
        rank_ic_12_ave = []
        for idx in range(12, len(rank_ic_ary)):
            rank_ic_12_ave.append(np.mean(rank_ic_ary[idx-12:idx+1]))
    
    
        fig, ax = plt.subplots()
        ax.bar(np.arange(1, len(rank_ic_ary)+1), np.asarray(rank_ic_ary) * 100, label='Rank IC')
        ax.plot(np.arange(12, len(rank_ic_ary)), np.asarray(rank_ic_12_ave) * 100, color='b', linewidth=3, label='12 peirod avg')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(-40, 40)
        ax.legend(loc='upper right')
        ax.set_xlabel('%s (from %d to %d)' % (self.period, self.start_year, self.end_year))
        
        t_stat = ss.ttest_1samp(rank_ic_ary, 0)
        plt.title('Rank ICs (t-stat=%.2f, ic=%.4f)' % (t_stat.statistic, np.mean(rank_ic_ary)))
        
        fig.savefig(self._filename('rank_ic'))
        plt.clf()
        plt.close()

    def draw_decay_profile(self, ave_lag_ic, success_rate):
        fig, ax1 = plt.subplots() 
        ax2 = plt.twinx()
        bar = ax1.bar(np.arange(1, len(ave_lag_ic)+1), np.asarray(ave_lag_ic) * 100, label='Ave(IC)')
        ax1.legend(loc='upper left')
        line = ax2.plot(np.arange(1, len(ave_lag_ic)+1), np.asarray(success_rate) * 100, color='b', linewidth=3, label='Success Rate')
        ax2.legend(loc='upper right')
        
        ax1.set_xlabel('lag (month)')
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_ylim(-10, 10)
        ax2.set_ylim(20, 80)
        ax1.tick_params(axis='y', colors='r') 
        ax2.tick_params(axis='y', colors='b') 
        plt.title('IC decay profile')
        
        fig.savefig(self._filename('decay_profile'))
        plt.clf()
        plt.close()

    def draw_fractile(self, sim_return):
        fig, ax = plt.subplots()

        upper_y = np.max(sim_return) + 100

        for line_idx in range(sim_return.shape[0]):
            label = str(line_idx + 1) if line_idx < sim_return.shape[0] - 1 else "Market"
            color = (line_idx / sim_return.shape[0], 0, 0)
            ax.plot(np.arange(0, sim_return.shape[1]), sim_return[line_idx], color=color, linewidth=3, label=label)
            ax.set_ylim(0, upper_y)
        ax.legend(loc='upper left')
        ax.set_xlabel('%s (from %d to %d)' % (self.period, self.start_year, self.end_year))
        
        plt.title('Fractile')
        
        fig.savefig(self._filename('fractile'))
        plt.clf()
        plt.close()

    def show_fractile_table(self, metrics_list):
        attr_list = ["total_return", "active_return", "tracking_error", "information_ratio", "ir_t_stat", "monthly_success_rate", "turnover", "volatility", "sharp_ratio", "sr_t_stat", "beta", "alpha"]
        header_width = 20
        col_width = 15
        # print table header
        print(''.ljust(header_width), end="")
        for m_idx in range(len(metrics_list)):
            if m_idx != len(metrics_list) - 1:
                print(("Fractile %d" % (m_idx + 1)).rjust(col_width), end="")
            else:
                print("Market".rjust(col_width), end="")
        print("")

        for attr_name in attr_list:
            print(attr_name.ljust(header_width), end="")
            for metrics in metrics_list:
                val = getattr(metrics, attr_name) or ""
                val_str = "%.2f" % val if type(val) != str else val
                print(val_str.rjust(col_width), end="")
            print("")
            

class Metrics:
    def __init__(self, return_ary, market_metrics=None, years=None):

        self.return_ary = return_ary

        self.total_return = np.mean(return_ary) * 12
        self.volatility = np.std(return_ary) * np.sqrt(12)
        self.sharp_ratio = self.total_return / self.volatility

        if market_metrics != None:
            active_return_ary = self.return_ary - market_metrics.return_ary
            self.active_return = np.mean(active_return_ary) * 12
            self.tracking_error = np.std(active_return_ary) * np.sqrt(12)
            self.information_ratio = self.active_return / self.tracking_error
            self.ir_t_stat = self.information_ratio * np.sqrt(years) if years != None else None
            self.monthly_success_rate = len([e for e in active_return_ary if e > 0]) / len(active_return_ary)
            self.sr_t_stat = (self.sharp_ratio - market_metrics.sharp_ratio) / (np.sqrt(2 / years))

            self.turnover = None
            self.beta = None
            self.alpha = None
        else:
            self.active_return = None
            self.tracking_error = None
            self.information_ratio = None
            self.ir_t_stat = None
            self.monthly_success_rate = None
            self.sr_t_stat = None
            self.turnover = None
            self.beta = None
            self.alpha = None
