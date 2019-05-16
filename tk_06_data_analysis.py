#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/raw_data/raw_data1.csv")
# 去掉功率为0的点
data = data[data["active_power"] > 1]
print(data.head(100))


# 绘制直方图
# f = plt.figure()
# f.add_subplot(1, 4, 1)
# sns.distplot(data["active_power"], bins=10)
# f.add_subplot(1, 4, 2)
# sns.distplot(data["wind_speed"], bins=10)
# f.add_subplot(1, 4, 3)
# sns.distplot(data["Generator_speed"], bins=10)
# f.add_subplot(1, 4, 4)
# sns.distplot(data["Gearbox_oil_temperature"], bins=10)


# 绘制相关图 蓝色正相关,红色负相关,白色无关
sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap=sns.color_palette("RdBu", n_colors=128))

# 绘制散点图
# sns.pointplot(x="wind_speed", y="active_power", data=data.head(10000), scale=0.25, join=False)



plt.show()