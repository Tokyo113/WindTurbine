#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/k_Means_15.csv")
print(data.groupby("category").count())
# print(data.head(20))


# 离散点阈值   (3, 1.97)  (15, 1.95)
threshold = 1.97



# 正常点与离群点
outier = data[data["distance"] >= threshold]
normal = data[data["distance"] < threshold]

# 去掉离群的簇  category = 14, 6, 10
cluster_15 = normal[(normal["category"] != 14) & (normal["category"] != 10)]
cluster_15 = cluster_15[(cluster_15["category"] != 6) | (cluster_15["active_power"] >2000)]

cluster_15_outier = cluster_15[cluster_15["category"] == 14]

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# f = plt.figure()
# f.add_subplot(1, 2, 1)
# 原始数据: 正常点与离群点
# plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
# plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
# plt.scatter(cluster_15_outier["wind_speed"], cluster_15_outier["active_power"], c='b', s=3, alpha=.5)

# f.add_subplot(1, 2, 2)
# 去掉离群点后的图像
plt.scatter(cluster_15["wind_speed"], cluster_15["active_power"], c='g', s=3, alpha=.5)

plt.show()


'''
5.21进展
1.K_Means聚类  k=15 去掉了离群点,可以和原图对比一下

思考:
1.下一步工作:考虑四分位法进一步剔除离群点,或者先用四分位法,再用K聚类
2.二次聚类,再聚类一次,剔除离群点
3.学习DBSCAN
4.其他特征的数据预处理

'''