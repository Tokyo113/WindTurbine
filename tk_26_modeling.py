#coding:utf-8
'''
@Time: 2019/8/9 下午1:55
@author: Tokyo
@file: tk_26_modeling.py
@desc: 对比三种模型和stacking模型
'''

import pandas as pd
import numpy as np
from tk_18_data_preprocessing import wt_preprocessing
import matplotlib.pyplot as plt
from tk_tools import WT_modeling
import matplotlib.dates as mdate
from tk_19_feature_selection import feature_selection
from tk_20_stacking_model import stacking_model, stacking_model2

df1 = pd.read_csv('./data/C/train/training set.csv')
df1 = df1.dropna()

df1 = feature_selection(df1)
features, label, names = wt_preprocessing(df1, False)
# stacking_model(features, label)
# WT_modeling(features, label)
df1_t = pd.read_csv('./data/C/hfj061_28/validation set.csv')
df1_t = df1_t.dropna()
df1_t = feature_selection(df1_t)
x_val, y_val, val_names = wt_preprocessing(df1_t, False)

y_test1, y_pred1 = stacking_model2(features, label, x_val, y_val)

y_pred1 = pd.Series(y_pred1)
y_pred1.plot(c='g', label='Predict')
y_test1 = y_test1.reset_index(drop=True)
# # flatten 降维
y_test1.plot(c='y', label='Test')
plt.show()




# df2 = pd.read_csv('./data/B/train/training set.csv')
# df2 = df2.dropna()
# df2 = feature_selection(df2)
# features, label, names = wt_preprocessing(df2, False)
# stacking_model(features, label)
# WT_modeling(features, label)