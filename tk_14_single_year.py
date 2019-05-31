#coding:utf-8
'''
@Time: 2019/5/31 下午2:06
@author: Tokyo
@file: tk_14_single_year.py
@desc:取一整年的数据进行分析
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('./data/year/raw_data2017.csv')

# 处理前的散点图
# plt.scatter(data["wind_speed"], data["active_power"], s=3, alpha=.5)


