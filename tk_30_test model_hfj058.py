#coding:utf-8
'''
@Time: 2019/8/12 下午8:54
@author: Tokyo
@file: tk_30_test model_hfj058.py
@desc:
'''


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tk_tools import WT_modeling
import matplotlib.dates as mdate
from tk_19_feature_selection import feature_selection
from tk_20_stacking_model import stacking_model, stacking_model2
from tk_27_preprocessing import wt_preprocessing
from tk_21_test_model57 import stacking_MD
from tk_tools import wt_Cusum_change_point_detection
df1 = pd.read_csv('./data/C/train/training set.csv')
df1 = df1.dropna()
df1 = feature_selection(df1)

df1_t = pd.read_csv('./data/C/test/hfj058_test_25/test data2.csv')
df1_t = df1_t.dropna()
df1_t = feature_selection(df1_t)

features, label, data_te, label_te = wt_preprocessing(df1, df1_t, False)

y_test1, y_pred1 = stacking_model2(features, label, data_te, label_te)

md1 = stacking_MD(features, label, data_te, label_te)
wt_Cusum_change_point_detection(md1, 1000, 0.99)
# 计算Cusum序列
arr = np.array(md1)
s = np.zeros(len(arr) + 1)
for i in range(1, len(arr)):
    s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
s = pd.Series(s)

# 作图
f = plt.figure(figsize=(12, 12))
ax1 = f.add_subplot(3, 1, 1)
y_pred1 = pd.Series(y_pred1)
y_pred1.plot(c='g', label='Predict')
y_test1 = y_test1.reset_index(drop=True)
# # flatten 降维
y_test1.plot(c='y', label='Test')

ax2 = f.add_subplot(3, 1, 2)
md1.plot()

ax3 = f.add_subplot(3, 1, 3)
s.plot()
plt.show()

