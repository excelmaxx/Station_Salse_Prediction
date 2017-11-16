#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

cmd = 'spark-submit --queue bdp_hecate_ipc --master yarn-client --num-executors 25 --executor-memory 20G --executor-cores 2 predict_sta.py'
rst = os.system(cmd)
if rst != 0:
    raise Exception(1)
