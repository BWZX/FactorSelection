import os
import pickle

def load_data(filename, start_year, end_year):
    # load of get data from interface
    if os.path.isfile(filename):
        f = open(filename, 'rb')
        month_data_ary = pickle.load(f)
    else:
        month_data_ary = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                date_str = "%d-%02d" % (year, month)
                print(date_str)
                cur_month_data = opdata.get_month(date_str)
                month_data_ary.append(cur_month_data)
    
        f = open(filename, 'wb')
        pickle.dump(month_data_ary, f)
    return month_data_ary
