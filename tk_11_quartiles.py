#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
data: 5.23
author: Tokyo
'''


data = pd.read_csv("./data/data_after_KMeans.csv")
data1 = data.drop("Unnamed: 0", axis=1)
print(len(data))


# 分桶
quartiles = pd.cut(data["wind_speed"], 20)
print(quartiles.head())


def get_status(group):
    q_interval = group.quantile(q=0.75) - group.quantile(q=0.25)
    high = group.quantile(q=0.75) + 1.5*q_interval
    low = group.quantile(q=0.25) - 1.5*q_interval

    return {'q_high': high, 'q_low': low,
            'count': group.count(), 'mean': group.mean()}




grouped = data["active_power"].groupby(quartiles).apply(get_status)

# print(grouped.apply(get_status))
print(type(grouped))
s = []
for a, b in data["active_power"].groupby(quartiles):
    label = []
    q_interval = b.quantile(q=0.75) - b.quantile(q=0.25)
    high = b.quantile(q=0.75) + 1.5 * q_interval
    low = b.quantile(q=0.25) - 1.5 * q_interval
    for i in b:
        if (i < low) | (i > high):
            label.append(1)
        else:
            label.append(0)
    label_np = np.array(label)

    tk = pd.concat([b, pd.Series(label_np, index=b.index)], axis=1)
    s.append(tk)

s = pd.concat(s)
s = s.drop("active_power", axis=1)

data = pd.concat([data1, s], axis=1)
data.columns = list(data1.columns) + ["outier"]
print(data.head())





