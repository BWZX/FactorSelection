import os
from opdata import opdata
import pickle

from cfgs.config import cfg

def _load_data(filename, start_year, end_year):
    # load of get data from interface
    filepath = os.path.join(cfg.data_dir, filename)
    if os.path.isfile(filepath):
        f = open(filepath, 'rb')
        month_data_ary = pickle.load(f)
    else:
        month_data_ary = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                date_str = "%d-%02d" % (year, month)
                print(date_str)
                cur_month_data = opdata.get_month(date_str)
                month_data_ary.append(cur_month_data)
    
        f = open(filepath, 'wb')
        pickle.dump(month_data_ary, f)
    return month_data_ary

def get_next_month(cur_time):
    '''
    Return the next month in the format of 'yyyy-mm'
    Params:
        cur_time: if the format of 'yyyy-mm-dd'
    '''
    year, month, day = [int(e) for e in cur_time.split('-')]
    if month == 12:
        next_month = "%d-%02d" % (year + 1, 1)
    else:
        next_month = "%d-%02d" % (year, month + 1)
    return next_month

def get_month_diff(time1, time2):
    '''
    Return the number of months between time1 and time2
    Params:
        time1, time2: in the format of 'yyyy-mm' or 'yyyy-mm-dd'
    '''
    time1_ary = [int(e) for e in time1.split('-')]
    time2_ary = [int(e) for e in time2.split('-')]

    year1 = time1_ary[0]
    month1 = time1_ary[1]

    year2 = time2_ary[0]
    month2 = time2_ary[1]

    tot_month1 = year1 * 12 + month1
    tot_month2 = year2 * 12 + month2

    return tot_month1 - tot_month2

def load_data(filename, start_month, end_month, stock_list='hs300', period='1m'):
    filepath = os.path.join(cfg.data_dir, filename)
    if os.path.isfile(filepath):
        f = open(filepath, 'rb')
        data = pickle.load(f)
        return data

    data = []
    while True:
        cur_data = opdata.get_all(stock_list, period, start_month)
        cur_end_time = cur_data[1]
        month_diff = get_month_diff(end_month, cur_end_time)
        if month_diff <= 0:
            if month_diff == 0:
                data += cur_data[0]
            else:
                data += cur_data[0][:month_diff]
            break
        start_month = get_next_month(cur_end_time)
        data += cur_data[0]

    f = open(filepath, 'wb')
    pickle.dump(data, f)
    return data
