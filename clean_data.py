import csv
from datetime import datetime
from pprint import pprint
import re
from mdd import *

Check_depth = 10 #number of lines till which you're willing to check for corruption
Flag_cols = ['flags', 'ecg_hr', 'spo2', 'ox_strength', 'ox_pulse', 'ibp1_sys', 'ibp1_dia', 'ibp1_mn', 'ibp1_hr', 'ibp2_sys', 'ibp2_dia', 'ibp2_mn', 'ibp2_hr', 'temp1_c', 'temp1_f', 'temp2_c', 'temp2_f', 'nibp_sys', 'nibp_dia', 'nibp_map', 'nibp_hr', 'etco2', 'inco2', 'ambient_pressure', 'co2_resp', 'date', 'time']
Date_cols = ['date', 'time', 'flags', 'ecg_hr', 'spo2', 'ox_strength', 'ox_pulse', 'ibp1_sys', 'ibp1_dia', 'ibp1_mn', 'ibp1_hr', 'ibp2_sys', 'ibp2_dia', 'ibp2_mn', 'ibp2_hr', 'temp1_c', 'temp1_f', 'temp2_c', 'temp2_f', 'nibp_sys', 'nibp_dia', 'nibp_map', 'nibp_hr', 'etco2', 'inco2', 'ambient_pressure', 'co2_resp']
Flag_re = re.compile(r'[0-9a-f]{8}')
Date_re = re.compile(r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}')
Date_time_fmt = '%m/%d/%Y-%I:%M:%S%p'

def check_goodness(filename):
    counts = [0, 0, 0]
    labels = ['flag','date','corrupt']
    with open(filename) as fd:
        i = 1
        for line in fd:
            x = line.split(',')
            first_term = x[0].strip()
            if Flag_re.match(first_term): counts[0]+=1
            elif Date_re.match(first_term): counts[1]+=1
            elif i>=Check_depth: break
            else:
                counts[2]+=1
                i += 1
    return labels[max([(counts[i], i) for i in range(3)])[1]]

def extract(csvfilename, infusiontime=None): #say for 14th August 2016 at 8:05 am, infusiontime = ('8/14/2016', '8:05:00 am'); note that spaces/case doesn't matter
    check_file = check_goodness(csvfilename)
    if infusiontime:
    	infusion_t = datetime.strptime(''.join('-'.join(infusiontime).split()), Date_time_fmt)
    	infused = True
    else:
    	infused = False
    if check_file == 'corrupt':
        print 'corrupt data file'
        return
    elif check_file == 'flag':
        print 'flag first type of file'
        fields = Flag_cols
    elif check_file == 'date':
        print 'date first type of file'
        fields = Date_cols
    data_path = csvfilename.split('/')
    data_path = data_path[:-1]
    data_path = '/'.join(data_path)+'/'
    data = multidimensional_data([('samples',[]),('vitals',[])], data_path)
    with open(csvfilename) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields)
        i = 1
        try:
            for row in reader:
                try:
                    current_t = datetime.strptime(''.join('-'.join([row['date'], row['time']]).split()), Date_time_fmt)
                    timestamp = current_t.strftime(Date_time_fmt)
                    if infused:
                        time_diff = current_t - infusion_t
                        time_diff_seconds = int(time_diff.total_seconds())
                        data.insert((timestamp, 'exp_time'), time_diff_seconds)
                        if time_diff_seconds<0:
                            time_diff_hrs_fmt = '-'+':'.join([str(-time_diff_seconds//3600), format(-time_diff_seconds//60%60, '02d'), format(-time_diff_seconds%60, '02d')])
                        else:
                            time_diff_hrs_fmt = ':'.join([str(time_diff_seconds//3600), format(time_diff_seconds//60%60, '02d'), format(time_diff_seconds%60, '02d')])
                        data.metadata('samples', timestamp, 'exp_time_hms', time_diff_hrs_fmt)
                    for key in row.keys():
                        if key not in [None, 'date', 'time', 'flags']:
                            data.insert((timestamp, key), float(row[key]))
                    i += 1
                except:
                	print 'corrupt row', i
                	i += 1
        except:
	        print 'strong corruption; cannot move beyond this point in file; exiting'
    return data

datafile = '../data/pigs/20160825_988_OR_vitals.txt'
data = extract(datafile, ('08/25/2016', '8:42:00am'))
data.writecsv('20160825_988_OR_vitals')