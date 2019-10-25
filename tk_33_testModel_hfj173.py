#coding:utf-8
'''
@Time: 2019/8/13 上午11:04
@author: Tokyo
@file: tk_33_testModel_hfj173.py
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

df1 = pd.read_csv('./data/B/train/training set.csv')
df1 = df1.dropna()
df1 = feature_selection(df1)

df1_t = pd.read_csv('./data/B/hfj173_test/data2018_April_test.csv')
df1_t = df1_t.dropna()
df1_t = feature_selection(df1_t)

features, label, data_te, label_te = wt_preprocessing(df1, df1_t, False)

y_test1, y_pred1 = stacking_model2(features, label, data_te, label_te)

md1 = stacking_MD(features, label, data_te, label_te)
# wt_Cusum_change_point_detection(md1, 1000, 0.99)
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
y_test1.plot(c='y', label='Measured value')
x_ticks = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax1.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21",  "Apr/27"],
                               rotation=30, fontsize=10)
plt.ylabel('Oil Temperature/(Deg.C)', fontsize=15)
plt.hlines(y=76, xmin=0, xmax=5800, colors='r', label='Upper Limit')
plt.legend(loc='lower left', ncol=3)
rect = plt.Rectangle((3600, 0), 150, 80, color='r', alpha=0.3)
ax1.add_patch(rect)
plt.annotate('Fault alarm', xy=(3704, 75), xycoords='data', fontsize=15,
                 xytext=(+40, -50), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))





ax2 = f.add_subplot(3, 1, 2)
md1.plot()
x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax2.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21",  "Apr/27"],
                               rotation=30, fontsize=10)
plt.ylabel('Mahalanobis Distance', fontsize=15)
y_ticks = ax2.set_yticks([0, 5, 10, 15, 20, 25])
rect = plt.Rectangle((3600, -2), 150, 80, color='r', alpha=0.3)
ax2.add_patch(rect)




ax3 = f.add_subplot(3, 1, 3)
s.plot()
x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax3.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21",  "Apr/27"],
                               rotation=30, fontsize=10)
plt.ylabel('CUSUM chart', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.annotate('Change Point 1', xy=(1178, s[1178]), xycoords='data', fontsize=15,
                 xytext=(0, +70), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate('Change Point 2', xy=(3704, s[3704]), xycoords='data', fontsize=15,
                 xytext=(+10, -180), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate('Change Point 3', xy=(1906, s[1906]), xycoords='data', fontsize=15,
                 xytext=(+10, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate('Change Point 4', xy=(2840, s[2840]), xycoords='data', fontsize=15,
                 xytext=(-15, -100), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
rect = plt.Rectangle((3600, -650), 150, 3700, color='r', alpha=0.3)
ax3.add_patch(rect)



plt.savefig('./data/figure/fault 173_1.png', dpi=400)
plt.show()

