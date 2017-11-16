#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
import numpy as np
import pandas as pd
import copy
from datetime import datetime, timedelta
import xgboost as xgb
import math
import os

conf = SparkConf().setAppName('cangchu_sta_train_daily')
conf.set('spark.default.parallelism', '2000')
sc = SparkContext(conf=conf)


sc.setLogLevel("ERROR")
def get_diff_days(date_str, diff=7):
    return datetime.strftime(datetime.strptime(date_str, date_format) + timedelta(diff), date_format)


date_format = '%Y-%m-%d'

stable_days_diff = 7
fr_start_date_str = datetime.strftime(datetime.today(), date_format)
# fr_start_date_str = datetime.strftime(datetime.today() - timedelta(2), date_format)
start_date_str = get_diff_days(fr_start_date_str, -(28 + stable_days_diff))
# threshold for new station
threshold_data_len = 365 + 30
threshold_date_str = get_diff_days(start_date_str, -threshold_data_len)

data_path = '/user/mart_dm_tbi/app.db/app_sfs_ord_history_time_series/source_type=station_ords_create_dt/date_type=day'
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


new_data = sc.textFile(data_path).map(new_split_data_and_rowkey).filter(lambda x: (x[col.index('deliver_type')] == u'自营') and (x[col.index('ord_type')] == u'普通')).persist()
new_std_sta_city_data = sc.textFile(std_sta_city_path).map(lambda x: x.split('#')).collect()


def get_date_slice(start_date_str, shift=0):
    cycle_len = 365
    pre_end_days = get_diff_days(start_date_str, diff=-(1 + shift))
    pre_days = get_diff_days(pre_end_days, diff=-window_len)
    his_pre_end_days = get_diff_days(pre_end_days, -cycle_len)
    his_pre_days = get_diff_days(pre_days, -cycle_len)
    his_days = get_diff_days(his_pre_days, diff=window_len)
    his_end_days = get_diff_days(his_pre_end_days, diff=window_len)
    return [pre_days, pre_end_days, his_days, his_end_days, his_pre_days, his_pre_end_days]


def new_get_pos_slice(line, start_date_str, shift=0):
    date_slice = get_date_slice(start_date_str, shift + 7)   # feature date
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
    # 如果是新站, 用std_num 的历史数据代替 cur_sta的历史数据
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
        cur_his_ts_qttys = std_sta_data[2].split(', ')
        std_his_data_pre = cur_his_ts_qttys[(pos_slice_list[0] + stas_date_shift):(pos_slice_list[1] + stas_date_shift)]
        std_his_data_his = cur_his_ts_qttys[(pos_slice_list[2] + stas_date_shift):(pos_slice_list[3] + stas_date_shift)]
        std_his_data_his_pre = cur_his_ts_qttys[(pos_slice_list[4] + stas_date_shift):(pos_slice_list[5] + stas_date_shift)]
        cur_his_data_pre = one_record[col.index('his_ts_qttys')].split(', ')[pos_slice_list[0]:pos_slice_list[1]]
        # 历史数据不足30天的新站, 不预测
        if len(cur_his_data_pre) < window_len:
            return None
        divisor = np.mean([(int(x) / float(y)) if float(y) > 0 else 1 for x in std_his_data_pre for y in cur_his_data_pre])
        rst = []
        rst.append(cur_his_data_pre)
        rst.append([x / divisor if isinstance(x, int) else x for x in std_his_data_his])
        rst.append([x / divisor if isinstance(x, int) else x for x in std_his_data_his_pre])
    rst = reduce(lambda x, y: x + y, rst)
    # 目标站点/ 不能凑足
    if len(rst) != 90:
        return None
    rst += [int(float(one_record[col.index('province_id')])), int(float(one_record[col.index('city_id')])), int(float(one_record[col.index('county_id')]))]
    return one_record + [rst]


def new_slice_label_data(line):
    pos_slice_list = map(int, map(float, line[col.index('pos_slice_list')].split(',')))
    rst = line[col.index('his_ts_qttys')].split(', ')[pos_slice_list[6]:pos_slice_list[6] + 1]
    rst = int(rst[0]) if (len(rst) == 1) & (rst != ['\N']) else 0
    return line + [rst]


def new_is_all_positive(line, idx):
    test_list = map(float, line[idx].split(','))
    for x in test_list:
        if x <= 0:
            return False
    return True


def get_model_date_str():
    today = datetime.today()
    weekday = today.weekday()
    sunday = today + timedelta(6 - weekday)
    return datetime.strftime(sunday, date_format)

global fr_start_date_str, start_date_str
today_date_str = [fr_start_date_str, copy.deepcopy(start_date_str)]
for label_feature_shift in xrange(fr_len):
    train_data_list = []
    for iter_shift in xrange(iter_count):
        start_date_str = get_diff_days(today_date_str[1], iter_shift)
        new_province_data = new_data.map(lambda x: new_get_pos_slice(x, start_date_str, label_feature_shift))\
                                    .filter(lambda x: new_is_all_positive(x, col.index('pos_slice_list')))
        new_province_data = new_province_data.map(new_slice_his_data).filter(lambda x: x is not None)
        new_province_data = new_province_data.map(new_slice_label_data).filter(lambda x: x[-1] > 0)
        train_data_list.append(new_province_data)
    total_train_data = sc.union(train_data_list)
    feature_data = np.array(total_train_data.map(lambda x: x[col.index('his_ts_qttys_slice')]).collect())
    label_data = np.array(total_train_data.map(lambda x: x[col.index('label_data')]).collect())
    xgmat = xgb.DMatrix(feature_data, label=label_data)
    params = {
        'num_round': 30,
        'subsample': 0.7,
        'silent': 1,
        'task': 'regression',
        'colsample_bytree': 0.8,
        'gamma': 1.1,
        'eta': 0.1,
        'objective': 'reg:linear',
        'max_depth': 15,
        'min_child_weight': 2.0,
        'eval_metric': 'rmse'
    }
    province_model = xgb.train(list(params.items()), xgmat, 1000, [])
    # model_path = '/home/songteng/sta/model/{}'.format(str(fr_start_date_str))
    model_path = '/data0/cmo_ipc/sfs/tengsong/sta/model/{}'.format(get_model_date_str())
    os.system("mkdir {}".format(model_path))
    model_file = '{0}/xgb_model_{1}'.format(model_path, label_feature_shift)
    os.system("rm -r {}".format(model_file))
    province_model.save_model(model_file)
