#coding:utf-8
'''
@Time: 2019/8/13 上午10:29
@author: Tokyo
@file: tk_31_typeB.py
@desc:
'''



from tk_28_saveModel import save_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tk_19_feature_selection import feature_selection
from tk_20_stacking_model import stacking_model, stacking_model2
from tk_27_preprocessing import wt_preprocessing
from tk_21_test_model57 import stacking_MD
from tk_tools import WT_modeling2

df1 = pd.read_csv('./data/B/train/training set.csv')
df1 = df1.dropna()

df1 = feature_selection(df1)

# stacking_model(features, label)

df1_t = pd.read_csv('./data/B/hfj149_83/test data.csv')
df1_t = df1_t.dropna()
df1_t = feature_selection(df1_t)

features, label, data_te, label_te = wt_preprocessing(df1, df1_t, False)
# save_model(features, label)

# 比较B类不同单一模型的效果
# WT_modeling2(features, label, data_te, label_te)
# 绘制验证正常模型的图
y_test1, y_pred1 = stacking_model2(features, label, data_te, label_te)
md1 = stacking_MD(features, label, data_te, label_te)
# 计算Cusum序列
arr = np.array(md1)
s = np.zeros(len(arr) + 1)
for i in range(1, len(arr)):
    s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
s = pd.Series(s)

# 作图
f = plt.figure(figsize=(10, 10))

ax1 = f.add_subplot(3, 1, 1)
y_pred1 = pd.Series(y_pred1)
y_pred1.plot(c='g', label='Predict')
y_test1 = y_test1.reset_index(drop=True)
# # flatten 降维
y_test1.plot(c='y', label='Measured value')
x_ticks = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax1.set_xticklabels(["Dec/01", "Dec/06", "Dec/11", "Dec/17", "Dec/22",  "Dec/29"],
                                   rotation=30, fontsize="small")
plt.ylabel('Oil Temperature/(Deg.C)')
plt.legend(loc='lower left', ncol=2)




ax2 = f.add_subplot(3, 1, 2)
md1.plot()
x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax2.set_xticklabels(["Dec/01", "Dec/06", "Dec/11", "Dec/17", "Dec/22",  "Dec/29"],
                                   rotation=30, fontsize="small")
plt.ylim((0, 20))
y_ticks = ax2.set_yticks([0, 5, 10, 15, 20])
plt.ylabel('Mahalanobis Distance')



ax3 = f.add_subplot(3, 1, 3)
s.plot()
x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
x_labels = ax3.set_xticklabels(["Dec/01", "Dec/06", "Dec/11", "Dec/17", "Dec/22",  "Dec/29"],
                                   rotation=30, fontsize="small")

plt.ylim((-1300, 200))
plt.ylabel('CUSUM chart')
plt.xlabel('Date')
plt.savefig('./data/figure/test normalB.png', dpi=300)
plt.show()

