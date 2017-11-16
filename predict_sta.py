#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyspark import SparkConf, SparkContext
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
import math
import os

conf = SparkConf().setAppName('cangchu_sta_predict_daily')
conf.set('spark.default.parallelism', '2000')
sc = SparkContext(conf=conf)


def get_diff_days(date_str, diff=7):
    return datetime.strftime(datetime.strptime(date_str, date_format) + timedelta(diff), date_format)


date_format = '%Y-%m-%d'
fr_start_date_str = datetime.strftime(datetime.today(), date_format)
stable_days_diff = 7
start_date_str = get_diff_days(fr_start_date_str, -(28 + stable_days_diff))

threshold_data_len = 365 + 30
threshold_date_str = get_diff_days(start_date_str, -threshold_data_len)

data_path = '/user/mart_dm_tbi/app.db/app_sfs_ord_history_time_series/source_type=station_ords_sta_dt/date_type=day'
# fr_len = 7
fr_len = 35
window_len = 30
iter_count = 28
dim_sta_path = '/user/mart_dm_tbi/dim.db/dim_sta'
std_sta_city_path = '/user/mart_dm_tbi/dim.db/city_std_sta'
ts_columns_names = ['tag', 'rowkey', 'date_unit', 'start_date', 'history_len', 'unit_len', 'time_span', 'his_ts_qttys', 'remark']
rowkey_columns_names = 'deliver_type, ord_type, region_name, area_id, area_name, sta_num, sta_name, province_id, city_id, county_id'.split(', ')
flatten_ts_columns_names = [ts_columns_names[0]] + rowkey_columns_names + ts_columns_names[2:]
dim_sta_columns = [
    'dim_sta_num', 'dim_sta_name', 'sta_num', 'dim_subd_num', 'dim_subd_name', 'subd_num', 'province_id', 'city_id', 'county_id', 'village_num', 'dim_sta_first_cate_cd', 'dim_sta_second_cate_cd',
    'trd_pty_sta_flag', 'sort_num', 'province_name', 'city_name', 'county_name', 'village_name', 'dim_sta_first_cate_desc', 'dim_sta_second_cate_desc', 'region_name', '211_flag', 'area_cd', 'area_name'
]
feature_list = 'deliver_type,ord_type,region_name,area_id,area_name,sta_num,sta_name,start_date,his_ts_qttys,province_id,city_id,county_id'.split(',')
feature_list = 'deliver_type,ord_type,sta_num,start_date,his_ts_qttys,province_id,city_id,county_id'.split(',')
exclude_stanum_feature_list = [x for x in feature_list if x != 'sta_num']
col = ['sta_num'] + exclude_stanum_feature_list + ['pos_slice_list', 'his_ts_qttys_slice', 'label_data']


def index(list_like, index_name):
    return list_like.index(index_name)


def new_split_data_and_rowkey(line):
    line = line.split('\x01')
    if line[1] is None:
        return None
    line[1] = line[1].split('$')
    line = [line[0]] + line[1] + line[2:]
    rst = []
    for x in ['sta_num'] + exclude_stanum_feature_list:
        rst.append(line[flatten_ts_columns_names.index(x)])
    return rst


def get_date_slice(start_date_str, shift=0):
    cycle_len = 365
    pre_end_days = get_diff_days(start_date_str, diff=-(1 + shift))
    pre_days = get_diff_days(pre_end_days, diff=-window_len)
    his_pre_end_days = get_diff_days(pre_end_days, -cycle_len)
    his_pre_days = get_diff_days(pre_days, -cycle_len)
    his_days = get_diff_days(his_pre_days, diff=window_len)
    his_end_days = get_diff_days(his_pre_end_days, diff=window_len)
    return [pre_days, pre_end_days, his_days, his_end_days, his_pre_days, his_pre_end_days]


def get_pos_slice(start_date_column_se, start_date_str, shift=0):
    date_slice = get_date_slice(start_date_str, shift)  # feature date
    date_slice += [get_diff_days(start_date_str, 0)]  # label date
    date_slice += [fr_start_date_str, get_diff_days(fr_start_date_str, fr_len)]  # real_date
    pos_slice = []
    for single_date in date_slice:
        pos_slice.append(start_date_column_se.apply(get_day_diff, args=(single_date, )))
    pos_slice = pd.concat(pos_slice, axis=1)
    pos_slice_list = pos_slice.apply(lambda x: ','.join([str(i) for i in x]), axis=1)
    return pos_slice_list


def new_get_pos_slice(line, start_date_str, shift=0):
    date_slice = get_date_slice(start_date_str, shift + 7)  # feature date
    date_slice += [get_diff_days(start_date_str, 0)]  # label data
    date_slice += [fr_start_date_str, get_diff_days(fr_start_date_str, fr_len)]
    pos_slice = []
    cur_start_date_str = line[col.index('start_date')]
    for single_date in date_slice:
        pos_slice.append(get_day_diff(cur_start_date_str, single_date))
    pos_slice = ','.join(map(str, pos_slice))
    return line + [pos_slice]


def get_day_diff(first_date_str, second_date_str=fr_start_date_str):
    first_date = datetime.strptime(first_date_str, date_format)
    second_date = datetime.strptime(second_date_str, date_format)
    return (second_date - first_date).days


def new_slice_his_data(one_record):
    std_sta_start_date = ""
    stas_date_shift = 0
    if one_record[col.index('start_date')] > threshold_date_str:
        cur_city_id = one_record[col.index('city_id')]
        cur_std_num = filter(lambda x: x[1] == cur_city_id, new_std_sta_city_data)
        if len(cur_std_num) == 0:
            return None
        else:
            cur_std_num = cur_std_num[0][0]
        std_sta_data = filter(lambda x: x[1] == cur_city_id, new_std_sta_city_data)[0][2]
        std_sta_start_date = filter(lambda x: x[1] == cur_city_id, new_std_sta_city_data)[0][3]
        cur_his_ts_qttys = std_sta_data.split(', ')
    else:
        cur_his_ts_qttys = one_record[col.index('his_ts_qttys')].split(', ')
    pos_slice_list = [int(float(x)) for x in one_record[col.index('pos_slice_list')].split(',')]
    rst = []
    counter = 0
    while counter < 6:
        if (pos_slice_list[counter] < 0) | (pos_slice_list[counter + 1] < 0):
            break
        cur_period_data = cur_his_ts_qttys[pos_slice_list[counter]:pos_slice_list[counter + 1]]
        if len(cur_period_data) == 0:
            break
        cur_period_data = map(int, cur_period_data)
        mean_value = np.mean(cur_period_data)
        std_value = np.std(cur_period_data)
        tmp_cur_period_data = [x for x in cur_period_data if (mean_value - 1 * std_value) < x < (1 * std_value + mean_value)]
        cur_mean_value = np.median(tmp_cur_period_data)
        if math.isnan(cur_mean_value):
            break
        cur_period_data = [x if (mean_value - std_value) < x < (std_value + mean_value) else cur_mean_value for x in cur_period_data]
        rst.append(cur_period_data)
        counter += 2
    # for new stations, use std_num replace cur_sta
    if len(rst) != 3:
        cur_city_id = one_record[col.index('city_id')]
        std_sta_data = filter(lambda x: x[1] == cur_city_id, new_std_sta_city_data)
        # 目标站点不存在
        if len(std_sta_data) == 0:
            return None
        else:
            std_sta_data = std_sta_data[0]
        cur_std_num = std_sta_data[0]
        std_sta_start_date = std_sta_data[3]
        stas_date_shift = get_day_diff(std_sta_start_date, one_record[col.index('start_date')])
        cur_his_ts_qttys = map(float, std_sta_data[2].split(', '))
        std_his_data_pre = cur_his_ts_qttys[(pos_slice_list[0] + stas_date_shift):(pos_slice_list[1] + stas_date_shift)]
        std_his_data_his = cur_his_ts_qttys[(pos_slice_list[2] + stas_date_shift):(pos_slice_list[3] + stas_date_shift)]
        std_his_data_his_pre = cur_his_ts_qttys[(pos_slice_list[4] + stas_date_shift):(pos_slice_list[5] + stas_date_shift)]
        cur_his_data_pre = map(float, one_record[col.index('his_ts_qttys')].split(', ')[pos_slice_list[0]:pos_slice_list[1]])
        # number of hisotry less than 30, do not predict
        if len(cur_his_data_pre) < window_len:
            return None
        # divisor = np.mean([(int(x) / float(y)) if float(y) > 0 else int(x) for x in std_his_data_pre for y in cur_his_data_pre])
        divisor = sum(std_his_data_his_pre) / float(sum(cur_his_data_pre)) if sum(cur_his_data_pre) != 0 else 0
        rst = []
        rst.append(cur_his_data_pre)
        rst.append([x / divisor if divisor != 0 else 1 for x in std_his_data_his])
        rst.append([x / divisor if divisor != 0 else 1 for x in std_his_data_his_pre])
    rst = reduce(lambda x, y: x + y, rst)
    rst = map(float, rst)

    if len(rst) != 90:
        return None
    rst += [one_record[col.index('province_id')], one_record[col.index('city_id')], one_record[col.index('county_id')]]
    return one_record + [rst]


def new_slice_real_data(one_record):
    pos_slice_list = [int(float(x)) for x in one_record[col.index('pos_slice_list')].split(',')]
    rst = one_record[col.index('his_ts_qttys')].split(', ')[pos_slice_list[7]:pos_slice_list[-1]]
    if len(rst) == fr_len:
        return one_record[col.index('sta_num')], map(int, rst)
    else:
        return None


def new_is_all_positive(line, idx):
    test_list = map(float, line[idx].split(','))
    for x in test_list:
        if x <= 0:
            return False
    return True


def get_model_date_str():
    base_day = datetime.today() - timedelta(7)
    weekday = base_day.weekday()
    sunday = base_day + timedelta(6 - weekday)
    return datetime.strftime(sunday, date_format)


def is_ready(line):
    sta_num = int(line[col.index('sta_num')])  # feature 1
    his_ts_qttys = map(int, line[col.index('his_ts_qttys')].split(', '))
    cur_start_date_str = line[col.index('start_date')]
    yesterday_str = datetime.strftime(datetime.today() - timedelta(1), date_format)
    date_gap = get_day_diff(cur_start_date_str, yesterday_str)
    if date_gap - len(his_ts_qttys) > 90:
        return False
    else:
        return True


new_data = sc.textFile(data_path).map(new_split_data_and_rowkey).filter(lambda x: (x[col.index('deliver_type')] == u'自营') and (x[col.index('ord_type')] == u'普通')).persist()


new_data = new_data\
          .filter(is_ready)\
          .map(lambda x: (x[0], x))\
          .reduceByKey(lambda x, y: x if len(x[4].split(', ')) > len(y[4].split(', ')) else y)\
          .map(lambda x: x[1]).persist()

new_std_sta_city_data = sc.textFile(std_sta_city_path).map(lambda x: x.split('#')).collect()

new_province_data = new_data.map(lambda x: new_get_pos_slice(x, fr_start_date_str))
new_province_data = new_province_data.map(new_slice_his_data).filter(lambda x: x is not None)

new_feature_data = np.array(new_province_data.map(lambda x: x[col.index('his_ts_qttys_slice')]).collect())
new_feature_sta_list = new_province_data.map(lambda x: x[0]).collect()
new_fr_rst_dict = {}
xgmat = xgb.DMatrix(new_feature_data)

for i in xrange(fr_len):
    m = xgb.Booster()
    model_path = '/data0/cmo_ipc/sfs/tengsong/sta/model/{1}/xgb_model_{0}'.format(str(i), get_model_date_str())
    m.load_model(model_path)
    tmp_rst = m.predict(xgmat)
    for j in xrange(len(new_feature_sta_list)):  # index for sta_list and tmp_rst
        new_fr_rst_dict[new_feature_sta_list[j]] = new_fr_rst_dict.get(new_feature_sta_list[j], []) + [tmp_rst[j]]


def or_data_split_data_and_rowkey(line):
    line = line.split('\x01')
    if line[1] is None:
        return None
    line[1] = line[1].split('$')
    line = [line[0]] + line[1] + line[2:]
    return line

def is_or_ready(line):
    sta_num = int(line[flatten_ts_columns_names.index('sta_num')])  # feature 1
    his_ts_qttys = map(int, line[flatten_ts_columns_names.index('his_ts_qttys')].split(', '))
    cur_start_date_str = line[flatten_ts_columns_names.index('start_date')]
    yesterday_str = datetime.strftime(datetime.today() - timedelta(1), date_format)
    date_gap = get_day_diff(cur_start_date_str, yesterday_str)
    if date_gap - len(his_ts_qttys) > 90:
        return False
    else:
        return True

or_data = sc.textFile(data_path).map(or_data_split_data_and_rowkey).filter(
    lambda x: (x[flatten_ts_columns_names.index('deliver_type')] == u'自营') and (x[flatten_ts_columns_names.index('ord_type')] == u'普通')).persist()


or_data = or_data.filter(is_or_ready)\
          .map(lambda x: (x[6], x))\
          .reduceByKey(lambda x, y: x if len(x[-2].split(', ')) > len(y[-2].split(', ')) else y)\
          .map(lambda x: x[1]).persist()


def append_fr_rst(line):
    sta_num = line[flatten_ts_columns_names.index('sta_num')]
    if sta_num in new_fr_rst_dict:
        fr_rst = new_fr_rst_dict[sta_num]
    else:
        fr_rst = ''
    return line + [fr_rst]


rst = or_data.map(append_fr_rst).persist()

fr_type_dict = {u'Self': '2', u'FBP': '0', u'3rd': '1'}

order_type_dict = {u'Fresh2': '5', u'Fresh1': '4', u'Direct': '3', u'Normal': '2', u'Frozen': '1', u'DeepFrozen': '0'}

region_id_dict = {u'North': '1', u'South': '2', u'East': '3', u'Northwest': '4', u'Southwest': '5', u'Middle': '6', u'Northeast': '7'}


def sum_last_60(line):
    rst = []
    rst.append(line[flatten_ts_columns_names.index('sta_num')])
    rst.append(line[flatten_ts_columns_names.index('deliver_type')])
    rst.append(line[flatten_ts_columns_names.index('ord_type')])
    his_ts_qttys = map(int, line[flatten_ts_columns_names.index('his_ts_qttys')].split(', '))
    cur_start_date_str = line[flatten_ts_columns_names.index('start_date')]
    cur_start_date = datetime.strptime(cur_start_date_str, date_format)
    yesterday_date = datetime.today() - timedelta(1)
    total_len = (yesterday_date - cur_start_date).days
    his_ts_qttys += [0] * (total_len - len(his_ts_qttys))
    rst.append(str(sum(his_ts_qttys[-60:])))
    return rst[0], rst


other_data = sc.textFile(data_path).map(or_data_split_data_and_rowkey).filter(lambda x: (x[flatten_ts_columns_names.index('deliver_type')] != u'自营') or (x[flatten_ts_columns_names.index('ord_type')] != u'普通'))
other_data = other_data\
          .filter(is_or_ready)\
          .map(lambda x: ((x[6], x[1], x[2]), x))\
          .reduceByKey(lambda x, y: x if len(x[-2].split(', ')) > len(y[-2].split(', ')) else y)\
          .map(lambda x: x[1]).persist()
other_data_ratio = other_data.map(sum_last_60).persist()
other_data_fr = other_data.map(append_fr_rst).persist()
base_data = sc.textFile(data_path).map(or_data_split_data_and_rowkey).filter(lambda x: (x[flatten_ts_columns_names.index('deliver_type')] == u'自营') and (x[flatten_ts_columns_names.index('ord_type')] == u'普通'))
base_data = base_data\
          .filter(is_or_ready)\
          .map(lambda x: ((x[6], x[1], x[2]), x))\
          .reduceByKey(lambda x, y: x if len(x[-2].split(', ')) > len(y[-2].split(', ')) else y)\
          .map(lambda x: x[1]).persist()
base_data_ratio = base_data.map(sum_last_60).persist()
joined = other_data_ratio.leftOuterJoin(base_data_ratio).persist()


def format_joined_data(line):
    if line[1][1] is None:
        return None
    rst = [[], 0]
    rst[0].append(line[0])  # sta_num
    rst[0].append(fr_type_dict[line[1][0][1]])  # fr_type
    rst[0].append(order_type_dict[line[1][0][2]])  # order_type
    rst[1] = float(line[1][0][-1]) / float(line[1][1][-1]) if float(line[1][1][-1]) != 0 else 1  # ratio
    return tuple(rst[0]), rst[1]


ratio_data = joined.map(format_joined_data).filter(lambda x: x is not None).persist()


def format_other_data_fr(line):
    sta_num = line[flatten_ts_columns_names.index('sta_num')]
    fr_type = fr_type_dict[line[flatten_ts_columns_names.index('deliver_type')]]
    order_type = order_type_dict[line[flatten_ts_columns_names.index('ord_type')]]
    return (sta_num, fr_type, order_type), line


def ratio_apply(line):
    his_ts_qttys = map(float, line[1][0][16].split(', '))
    real_sum = sum(his_ts_qttys[-35:])
    ratio = line[1][-1]
    ratio = ratio if ratio != 1 else 0
    fr_rst = line[1][0][-1]
    fr_rst = [x * ratio for x in fr_rst]
    fr_sum = sum(fr_rst)
    if fr_sum > 2 * real_sum:
        fr_rst = [float(real_sum) / 35] * 35
    line[1][0][-1] = fr_rst
    return line[1][0]


other_data_fr_ratio = other_data_fr.map(format_other_data_fr).leftOuterJoin(ratio_data).map(ratio_apply).persist()
def format_as_rst(line):
    cur_start_date_str = line[flatten_ts_columns_names.index('start_date')]
    his_ts_qttys = map(int, line[flatten_ts_columns_names.index('his_ts_qttys')].split(', '))
    cur_start_date = datetime.strptime(cur_start_date_str, date_format)
    yesterday_date = datetime.today() - timedelta(1)
    real_data_len = (yesterday_date - cur_start_date).days + 1
    cur_his_data_len = len(his_ts_qttys)
    if real_data_len != cur_his_data_len:
        his_ts_qttys += [0] * (real_data_len - cur_his_data_len)
    first_part = ','.join(map(str, his_ts_qttys[-60:]))
    second_part = ','.join(map(str, line[-1][:28]))
    third_part_len = 7 - datetime.today().weekday()
    third_part = ','.join(map(str, line[-1][28:28 + third_part_len]))
    week_ord = []
    for i in range(1, 10):
        week_ord.append(sum(his_ts_qttys[:-10][-i - 7:-i]))
    week_ord = reversed(week_ord)
    forth_part = ','.join(map(str, week_ord))
    cur_values = '$'.join([first_part, second_part, third_part, forth_part])
    his_value = [''] * 4
    if real_data_len > 365:
        his_value[0] = ','.join(map(str, his_ts_qttys[-365 + 1 - 60:-365 + 1]))
        his_value[1] = ','.join(map(str, his_ts_qttys[-365 + 1:-365 + 28 + 1]))
        his_value[2] = ','.join(map(str, his_ts_qttys[-365 + 28 + 1:-365 + 28 + 1 + third_part_len]))
        his_week_ord = []
        for i in range(1, 10):
            his_week_ord.append(sum(his_ts_qttys[-365 + 60 + 28 + 1 + third_part_len - i - 7:-365 + 60 + 28 + 1 + third_part_len - i]))
        his_week_ord = reversed(his_week_ord)
        his_value[3] = ','.join(map(str, his_week_ord))
    his_values = '$'.join(his_value)
    line = line + [cur_values] + [his_values]
    rst = [''] * 15
    rst[1] = region_id_dict[line[flatten_ts_columns_names.index('region_name')]]
    rst[2] = line[flatten_ts_columns_names.index('region_name')]
    rst[3] = line[flatten_ts_columns_names.index('area_id')]
    rst[4] = line[flatten_ts_columns_names.index('area_name')]
    rst[5] = line[flatten_ts_columns_names.index('sta_num')]
    rst[6] = line[flatten_ts_columns_names.index('sta_name')]
    # rst[7] = ','.join(map(str, his_ts_qttys))
    # return '#'.join(rst)
    rst[7] = datetime.strftime(yesterday_date, date_format)
    rst[8] = datetime.strftime(yesterday_date + timedelta(7 - yesterday_date.weekday()), date_format)
    rst[9] = fr_type_dict[line[flatten_ts_columns_names.index('deliver_type')]]
    rst[10] = order_type_dict[line[flatten_ts_columns_names.index('ord_type')]]
    rst[11] = u'1'  # 妥投
    rst[12] = cur_values
    rst[13] = his_values
    return '#'.join(rst)


base_rst_data = rst.map(format_as_rst)

other_rst_data = other_data_fr_ratio.map(format_as_rst)

final_rst = base_rst_data.union(other_rst_data)

output_data_path = '/user/mart_dm_tbi/dev.db/cangchu_sta_ord_sta_rst/{}'.format(fr_start_date_str)
os.system('hadoop fs -rm -r {}'.format(output_data_path))
final_rst.repartition(5).saveAsTextFile(output_data_path, compressionCodecClass="org.apache.hadoop.io.compress.DefaultCodec")

