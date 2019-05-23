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
data = data.drop(["category", "distance"], axis=1)
data = data.drop("Unnamed: 0", axis=1)

# 参数初始化
# 聚类个数
k = 15
# 离散点阈值   (3, 1.97)  (15, 1.95)
threshold = 1.95
# 聚类最大循环次数
iteration = 500
# 数据标准化
data_zs = 1.0*(data - data.mean())/data.std()


# 只选择风速和功率
data_tk = data_zs[["wind_speed", "active_power"]]


# 聚类
from sklearn.cluster import KMeans
model = KMeans(n_clusters=k, max_iter=iteration)
model.fit(data_tk)
# print(pd.Series(model.labels_, index=data1.index))
# 添加类别属性列
cluster_data = pd.concat([data, pd.Series(model.labels_, index=data_zs.index)], axis=1)
cluster_data.columns = list(data.columns) + ["category"]
print(cluster_data.groupby("category").count())



# 计算相对距离
norm = []
for i in range(k):
    # norm_tmp = data_zs[["Generator_speed", "Rotor_speed", "Gearbox_oil_temperature",
    #              "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
    #              "wind_speed", "active_power"]][cluster_data["category"] == i]-model.cluster_centers_[i]
    # 只考虑风速和功率计算距离
    norm_tmp = data_zs[["wind_speed", "active_power"]][cluster_data["category"] == i] - model.cluster_centers_[i]
    # 求绝对距离
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)

    # 求相对距离
    norm.append(norm_tmp/norm_tmp.median())



norm = pd.concat(norm)

cluster_data = pd.concat([cluster_data, norm], axis=1)
cluster_data.columns = list(data_zs.columns) + ["category"] + ["distance"]
print(cluster_data.head())
# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 区别离群点
outier = cluster_data[cluster_data["distance"] >= threshold]
normal = cluster_data[cluster_data["distance"] < threshold]


# 三个类
cluster1 = cluster_data[cluster_data["category"] == 0]
cluster2 = cluster_data[cluster_data["category"] == 1]
cluster3 = cluster_data[cluster_data["category"] == 2]

# plt.scatter(cluster1["wind_speed"], cluster1["active_power"], c='r', s=3, alpha=.5)
# plt.scatter(cluster2["wind_speed"], cluster2["active_power"], c='b', s=3, alpha=.5)
# plt.scatter(cluster3["wind_speed"], cluster3["active_power"], c='g', s=3, alpha=.5)


# 正常点与离群点
plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
plt.show()



'''
二次聚类:
仍然使用风速和功率经标准化后建立K_Means模型,
分三类的话离群点识别效果并不好,但分成三类聚类效果不错
k=10,15时能够识别一部分离群点,效果还行,现在考虑四分位法去除离群点


'''