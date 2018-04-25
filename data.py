import os
from opdata import opdata
import pickle
import datetime

from cfgs.config import cfg

def get_next_month(cur_time):
    '''
    Return the next month in the format of 'yyyy-mm'
    Params:
        cur_time: in the format of 'yyyy-mm-dd'
    '''
    year, month, day = [int(e) for e in cur_time.split('-')]
    if month == 12:
        next_month = "%d-%02d" % (year + 1, 1)
    else:
        next_month = "%d-%02d" % (year, month + 1)
    return next_month

def get_next_day(cur_time):
    '''
    Return the next day in the format of 'yyyy-mm-dd'
    Params:
        cur_time: in the format of 'yyyy-mm-dd'
    '''
    date = datetime.datetime.strptime(cur_time, "%Y-%m-%d")
    date += datetime.timedelta(days=1)
    next_day = '{0:%Y-%m-%d}'.format(date)
    return next_day

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

def get_day_diff(time1, time2):
    '''
    Return the number of days between time1 and time2
    Params:
        time1, time2: in the format of 'yyyy-mm' or 'yyyy-mm-dd'
    '''
    # first convert to datetime
    time1 = get_next_month(time1 + "-01")
    time1 = datetime.datetime.strptime(time1, "%Y-%m")
    time2 = datetime.datetime.strptime(time2, "%Y-%m-%d")

    return (time1 - time2).days - 1

def load_data(filename, start_month, end_month, factor_list, stock_list='hs300', period='1m'):
    filepath = os.path.join(cfg.data_dir, filename)
    if os.path.isfile(filepath):
        f = open(filepath, 'rb')
        data = pickle.load(f)
        return data

    if period in ['1m', '3m', '6m']:
        start_date = start_month
    else:
        start_date = "%s-01" % start_month

    data = []
    while True:
        cur_data = opdata.get_all(stock_list, period, start_date, factor_list)
        cur_end_time = cur_data[1]
        if period in ['1m', '3m', '6m']:
            month_diff = get_month_diff(end_month, cur_end_time)
            if month_diff <= 0:
                if month_diff == 0:
                    data += cur_data[0]
                else:
                    data += cur_data[0][:month_diff]
                break
            start_date = get_next_month(cur_end_time)
        else:
            day_diff = get_day_diff(end_month, cur_end_time)
            if day_diff <= 0:
                # end_month (actually end_month with last day in the month) is before cur_end_time
                for dp in cur_data[0]:
                    dp_date = dp['date'][0]
                    if get_month_diff(end_month, dp_date) >= 0:
                        data.append(dp)
                    else:
                        break
                break
            start_date = get_next_day(cur_end_time)
        data += cur_data[0]

    f = open(filepath, 'wb')
    pickle.dump(data, f)
    return data
