#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("./data/raw_data/raw_data_state.csv").head(45000)
# 去掉功率为0的点
data = data[data["active_power"] > 0][data["state"] == 6]
# data =data[data["active_power"] > 1]
# print(data.groupby("state").count())
# 切片:每隔5min取样
data1 = data.iloc[0:35592:5].drop("state", axis=1)
data1 = data1.drop("date", axis=1)
# 一共6919条数据
# print(data1.head())

# 参数初始化
# 聚类个数
k = 3
# 离散点阈值
threshold = 1.9
# 聚类最大循环次数
iteration = 500
# 数据标准化
data_zs = 1.0*(data1 - data1.mean())/data1.std()


# 聚类
from sklearn.cluster import KMeans
model = KMeans(n_clusters=k, max_iter=iteration)
model.fit(data1)
# print(pd.Series(model.labels_, index=data1.index))
# 添加类别属性列
cluster_data = pd.concat([data1, pd.Series(model.labels_, index=data1.index)], axis=1)
cluster_data.columns = list(data1.columns) + ["category"]


norm = []
for i in range(k):
    # norm_tmp = cluster_data[["Generator_speed", "Rotor_speed", "Gearbox_oil_temperature",
    #              "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
    #              "wind_speed", "active_power"]][cluster_data["category"] == i]-model.cluster_centers_[i]
    # 只考虑风速和功率计算距离
    norm_tmp = cluster_data[["wind_speed", "active_power"]][cluster_data["category"] == i] - model.cluster_centers_[i][5:]
    # 求绝对距离
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
    print(norm_tmp.median())
    # 求相对距离
    norm.append(norm_tmp/norm_tmp.median())



norm = pd.concat(norm)
cluster_data = pd.concat([cluster_data, norm], axis=1)
cluster_data.columns = list(data1.columns) + ["category"] + ["distance"]


outier = cluster_data[cluster_data["distance"] >= threshold]
normal = cluster_data[cluster_data["distance"] < threshold]
cluster1 = cluster_data[cluster_data["category"] == 0]
cluster2 = cluster_data[cluster_data["category"] == 1]
cluster3 = cluster_data[cluster_data["category"] == 2]

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 正常点
# norm[norm <= threshold].plot(style='go')
# # 离群点
# discrete_points = norm[norm > threshold]
# discrete_points.plot(style='ro')
# print(discrete_points)
plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
# plt.scatter(cluster1["wind_speed"], cluster1["active_power"], c='r', s=3, alpha=.5)
# plt.scatter(cluster2["wind_speed"], cluster2["active_power"], c='b', s=3, alpha=.5)
# plt.scatter(cluster3["wind_speed"], cluster3["active_power"], c='g', s=3, alpha=.5)
# plt.show()


'''
聚类效果较好,但无法识别离群点
可能的问题:
1. 标准化问题,现在没有使用标准化
2. 尝试其他聚类方法?密度?
3. 聚类特征量太多了?只用风速和功率(但现在的聚类效果其实很好)



思考:
问题在于求相对距离的公式,最后除以中位数,中位数受极端值影响很大
'''



