import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.stats.stats import pearsonr
import scipy.stats as ss

import numpy as np

class Output:
    def __init__(self, factor_code, start_year, end_year):
        self.factor_code = factor_code
        self.start_year = start_year
        self.end_year = end_year

    def _filename(self, name):
        return '%s (%d-%d): %s.png' % (self.factor_code, self.start_year, self.end_year, name)

    def draw_rank_ic(self, rank_ic_ary):
        rank_ic_12_ave = []
        for idx in range(12, len(rank_ic_ary)):
            rank_ic_12_ave.append(np.mean(rank_ic_ary[idx-12:idx+1]))
    
    
        fig, ax = plt.subplots()
        ax.bar(np.arange(1, len(rank_ic_ary)+1), np.asarray(rank_ic_ary) * 100, label='Rank IC')
        ax.plot(np.arange(12, len(rank_ic_ary)), np.asarray(rank_ic_12_ave) * 100, color='b', linewidth=3, label='12m avg')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(-40, 40)
        ax.legend(loc='upper right')
        ax.set_xlabel('Monthly (from %d to %d)' % (self.start_year, self.end_year))
        
        t_stat = ss.ttest_1samp(rank_ic_ary, 0)
        plt.title('Rank ICs (t-stat=%.2f)' % t_stat.statistic)
        
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

