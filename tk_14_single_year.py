#coding:utf-8
'''
@Time: 2019/5/31 下午2:06
@author: Tokyo
@file: tk_14_single_year.py
@desc:取一整年的数据进行分析和预处理
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_preprocessing(filename):
    data = pd.read_csv(filename)

    # 选择状态为6的数据并去掉state列
    data = data[data["state"] == 6].drop("state", axis=1)

    # 去掉功率为0的数据
    data = data[data["active_power"] > 1]

    # 取每10min数据
    data = data[::10]

    print(data.describe())

    # 处理前的散点图
    # plt.scatter(data["wind_speed"], data["active_power"], s=3, alpha=.5)
    # plt.show()
    print(data.mean())
    return data


def K_Means(data):
    """
    后续考虑问题:
    1.如何选择聚类个数?
    2.如何确定阈值?
    3.聚类标准:目前是欧氏距离,用马氏距离?
    :param data: 数据集
    """
    from sklearn.cluster import KMeans
    # 参数初始化
    # 聚类个数
    k = 15
    # 离散点阈值   (3, 1.97)  (15, 1.95)
    threshold = 1.95
    # 聚类最大循环次数
    iteration = 500
    # 数据标准化  z-score
    # 马氏距离? 博客标准化的缺陷
    data1 = data.drop("date", axis=1)
    data_zs = 1.0 * (data1 - data1.mean()) / data1.std()

    # 只使用风速和功率进行聚类
    data_tk = data_zs[["wind_speed", "active_power"]]

    model = KMeans(n_clusters=k, max_iter=iteration, )
    model.fit(data_tk)

    # 添加类别属性列
    cluster_data = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    cluster_data.columns = list(data.columns) + ["category"]
    # print(cluster_data.groupby("category").count())

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
        norm.append(norm_tmp / norm_tmp.median())

    norm = pd.concat(norm)
    cluster_data = pd.concat([cluster_data, norm], axis=1)
    cluster_data.columns = list(data.columns) + ["category"] + ["distance"]
    return cluster_data


def DBSCAN_cluster(data):
    """
    效果还行,收敛时间较长
    基本可以识别离群点
    :param data:
    """
    from sklearn.cluster import DBSCAN
    # (0.2, 250)
    eps = 0.2
    minPts = 250
    # 数据标准化  z-score
    # 马氏距离? 博客标准化的缺陷
    data1 = data.drop("date", axis=1)
    data_zs = 1.0 * (data1 - data1.mean()) / data1.std()

    # 只使用风速和功率进行聚类
    data_tk = data_zs[["wind_speed", "active_power"]]
    model = DBSCAN(eps=eps, min_samples=minPts, algorithm='kd_tree')
    model.fit(data_tk)

    # 添加类别属性列
    cluster_data = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    cluster_data.columns = list(data.columns) + ["category"]
    print(cluster_data.groupby("category").count())

    cluster_data["category"][(cluster_data["category"] == -1) & (cluster_data["active_power"] > 2000)] = 0
    outier = cluster_data[(cluster_data["category"] == -1)]
    normal = cluster_data[cluster_data["category"] != -1]
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 正常点与离群点
    # plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
    # plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
    #
    # plt.show()

    normal = normal.drop("category", axis=1)
    print(normal.describe())
    return normal


def draw_clusters(cluster_data):
    """
    K_Means聚类后作图
    :param cluster_data:
    """
    threshold = 2.5
    outier = cluster_data[cluster_data["distance"] >= threshold]
    normal = cluster_data[cluster_data["distance"] < threshold]
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 正常点与离群点
    plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
    plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
    plt.show()


def Quartiles(data):
    data1 = data.drop("date", axis=1)
    column_name = list(data.columns)
    # 分桶
    quartiles = pd.cut(data["wind_speed"], 20)

    # 四分位法处理数据
    s = []
    k = 3
    for a, b in data["active_power"].groupby(quartiles):
        label = []
        q_interval = b.quantile(q=0.75) - b.quantile(q=0.25)
        high = b.quantile(q=0.75) + k * q_interval
        low = b.quantile(q=0.25) - k * q_interval
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

    data = pd.concat([data, s], axis=1)
    data.columns = column_name + ["outlier"]

    # 绘制图像
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 剔除离群点
    normal = data[data["outlier"] == 0]
    outlier = data[data["outlier"] == 1]

    # 正常点与离群点
    plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
    plt.scatter(outlier["wind_speed"], outlier["active_power"], c='r', s=3, alpha=.5)
    plt.show()

    normal = normal.drop("outlier", axis=1)
    print(normal.describe())
    return normal

def wt_preprocessing(filename, gs=False, rs=False, dt=False, ndt=False, ws=False, ap=False):
    """
    对各列特征进行标准化或者归一化处理
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # 1.读入数据
    df = pd.read_csv(filename)
    # 得到标签(待预测值)
    label = df["Gearbox_oil_temperature"]

    df = df.drop("Gearbox_oil_temperature", axis=1)

    # 特征处理
    scaler_lst = [gs, rs, dt, ndt, ws, ap]
    column_lst = ["Generator_speed", "Rotor_speed",
                  "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
                  "wind_speed", "wind_speed"]

    for i in range(len(scaler_lst)):
        if scaler_lst[i]:
            df[column_lst[i]] = \
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

        else:
            df[column_lst[i]] = \
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

    return df, label


def WT_modeling():
    pass


def main():
    # 数据预处理
    filename = './data/year/raw_data2018.csv'
    raw_data = data_preprocessing(filename)
    normal_data = DBSCAN_cluster(raw_data)
    data_pre = Quartiles(normal_data)
    # cluster_data = K_Means(raw_data)
    # draw_clusters(cluster_data)
    # data_pre.to_csv('./data/year/data_pre2018.csv', index=None)

    # 建模
    # 标准化
    data_2017 = './data/year/data_pre2017.csv'
    data_2018 = './data/year/data_pre2018.csv'


if __name__ == '__main__':
    main()



'''
思考:
1.DBSCAN的参数如何选择?尝试k-dist方法,选择的依据
2.目前采用的是基于欧氏距离,效果还可以,尝试马氏距离




'''