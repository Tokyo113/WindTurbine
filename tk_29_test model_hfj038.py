#coding:utf-8
'''
@Time: 2019/8/12 下午8:43
@author: Tokyo
@file: tk_29_test model_hfj038.py
@desc:
stacking_model2 and stacking_MD 里使用的是保存的模型
验证集的图片注意调整y轴的范围,可以与故障测试集进行对比.
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

df1_t = pd.read_csv('./data/C/test/hfj038_test_5/test data1.csv')
df1_t = df1_t.dropna()
df1_t = feature_selection(df1_t)

features, label, data_te, label_te = wt_preprocessing(df1, df1_t, False)

y_test1, y_pred1 = stacking_model2(features, label, data_te, label_te)

md1 = stacking_MD(features, label, data_te, label_te)
# wt_Cusum_change_point_detection(md1, 1000, 0.90)
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
x_labels = ax1.set_xticklabels(["Mar/01", "Mar/07", "Mar/14", "Mar/24", "Mar/29",  "Apr/02"],
                                   rotation=30, fontsize=10)
plt.ylabel('Oil Temperature/(Deg.C)', fontsize=15)
plt.hlines(y=76, xmin=0, xmax=5600, colors='r', label='Upper Limit')
plt.legend(loc='lower left', ncol=3)
rect = plt.Rectangle((3950, 15), 220, 70, color='r', alpha=0.3)
ax1.add_patch(rect)
plt.annotate('Fault alarm', xy=(4220, 72), xycoords='data', fontsize=15,
                 xytext=(+70, -150), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

ax2 = f.add_subplot(3, 1, 2)
md1.plot()
x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax2.set_xticklabels(["Mar/01", "Mar/07", "Mar/14", "Mar/24", "Mar/29",  "Apr/02"],
                                   rotation=30, fontsize=10)
plt.ylabel('Mahalanobis Distance', fontsize=15)
y_ticks = ax2.set_yticks([0, 5, 10, 15, 20, 25])
rect = plt.Rectangle((3950, -2), 220, 27, color='r', alpha=0.3)
ax2.add_patch(rect)


ax3 = f.add_subplot(3, 1, 3)
s.plot()
x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax3.set_xticklabels(["Mar/01", "Mar/07", "Mar/14", "Mar/24", "Mar/29",  "Apr/02"],
                                   rotation=30, fontsize=10)
plt.ylabel('CUSUM chart', fontsize=15)
plt.xlabel('Date', fontsize=15)
plt.annotate('Change Point 1', xy=(2148, s[2148]), xycoords='data', fontsize=15,
                 xytext=(-30, +70), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate('Change Point 2', xy=(4015, s[4015]), xycoords='data', fontsize=15,
                 xytext=(+10, +40), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate('Change Point 3', xy=(3170, s[3170]), xycoords='data', fontsize=15,
                 xytext=(-10, +50), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))



rect = plt.Rectangle((3950, -1600), 220, 1800, color='r', alpha=0.3)
ax3.add_patch(rect)
plt.savefig('./data/figure/fault 038_1.png', dpi=400)
plt.show()

