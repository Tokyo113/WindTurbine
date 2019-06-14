#coding:utf-8
'''
@Time: 2019/6/14 下午2:00
@author: Tokyo
@file: tk_18_data_preprocessing.py
@desc:使用2018年七月一个月的数据,实现了两个特征选择的函数,
'''


import pandas as pd
import numpy as np
import seaborn as sns
from tk_14_single_year import Quartiles, DBSCAN_cluster, wt_preprocessing, wt_params
def wt_draw_scatter(data, x, y):
    """
    绘制散点图函数
    :param data: 数据
    :param x: 散点图x轴数据
    :param y: 散点图y轴数据
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(data[x], data[y], c='g', s=3, alpha=.5)
    plt.xlim((0, 18))
    plt.show()

def feature_RFE(data):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    data = data.drop("date", axis=1)
    data = data.drop("state", axis=1)

    # 得到特征和标签
    label = data["Gearbox_oil_temp"]
    features = data.drop("Gearbox_oil_temp", axis=1)
    names = features.columns.values

    lr = LinearRegression()

    rfe = RFE(lr, n_features_to_select=20)
    rfe.fit(features, label)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x:round(x, 4), rfe.ranking_), names)))
    print(rfe.n_features_)


def feature_filter(data):
    from sklearn.feature_selection import SelectKBest, f_regression
    data = data.drop("date", axis=1)
    data = data.drop("state", axis=1)


    # 得到特征和标签
    label = data["Gearbox_oil_temp"]
    features = data.drop("Gearbox_oil_temp", axis=1)
    names = features.columns.values

    fil = SelectKBest(f_regression, k=20)
    fil.fit(features, label)
    print(sorted(zip(map(lambda x: round(x, 4), fil.scores_), names), reverse=True))




def main():
    # df = pd.read_csv('./data/year/feature2018_31.csv').head(45000)
    # # 去掉功率为0的点
    # df= df[df["Active_power"] > 1][df["state"] == 6]
    # print(df.groupby("state").count())
    # # 切片:每隔5min取样
    # df = df.iloc[0:35592:5]
    # df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
    #
    # wt_draw_scatter(df, 'wind_speed', 'active_power')
    # df_1 = DBSCAN_cluster(df, 0.1, 55)
    # df_2 = Quartiles(df_1, 1.5, 80)
    # df_2.to_csv('./data/data2018_single_month.csv', index=None)
    data2018 = './data/data2018_single_month.csv'
    data = pd.read_csv(data2018)
    # wt_draw_scatter(data, 'wind_speed', 'active_power')
    feature_RFE(data)
    feature_filter(data)
    # X_tt, Y_tt = wt_preprocessing(data2018)
    # wt_params(X_tt, Y_tt)







if __name__ == '__main__':
    main()


