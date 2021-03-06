#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# data = pd.read_csv("./data/raw_data/raw_data_state.csv").head(45000)
# data = pd.read_csv("./data/year/feature2018_31.csv").head(45000)
df = pd.read_csv('./data/year/feature2018_38.csv')
# 去掉功率为0的点
# data = data[data["Active_power"] > 1][data["state"] == 6]
# data =data[data["active_power"] > 1]
# print(data.groupby("state").count())
# 切片:每隔5min取样
# data1 = data.iloc[0:35592:5]
# print(data1.head())
df = df.iloc[11::5]
plt.xlabel('Wind Speed/(m/s)')
plt.ylabel('Active Power/(kW)')

# sns.pointplot(x="wind_speed", y="active_power", data=data1.head(100), scale=0.25, join=False, ax=ax)
# 绘制散点图
plt.scatter(df["Wind_speed"], df["Active_power"], s=3, alpha=.5)
# plt.savefig('./data/figure/raw_data.png')
plt.show()