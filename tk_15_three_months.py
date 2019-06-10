#coding:utf-8
'''
@Time: 2019/6/10 下午5:27
@author: Tokyo
@file: tk_15_three_months.py
@desc: 18年前三个月建立正常模型,第四个月测试
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/year/data_pre2018.csv")
df = data[0:1100]

print(df)