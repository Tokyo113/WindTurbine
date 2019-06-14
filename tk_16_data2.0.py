#coding:utf-8
'''
@Time: 2019/6/13 上午11:38
@author: Tokyo
@file: tk_16_data2.0.py
@desc:
'''

import pandas as pd
import numpy as np
import seaborn as sns
from tk_14_single_year import DBSCAN_cluster,Quartiles,wt_preprocessing, wt_params


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




def main():
    # df = pd.read_csv('./data/year/data_pre2018.csv')
    # df = df[0:8786]
    # df = df[df['wind_speed'] <= 18][df["wind_speed"] >= 2.5]
    # df1 = pd.read_csv('./data/data_preprocessing.csv')
    # print(df1.describe())
    # wt_draw_scatter(df1, "wind_speed", "active_power")
    # 2017一整年数据处理
    # df_1 = DBSCAN_cluster(df, 0.1, 108)
    # df_1 = DBSCAN_cluster(df, 0.2, 150)
    # df_2 = Quartiles(df, 1.5, 500)
    # df_3 = DBSCAN_cluster(df_2, 0.19, 137)
    # wt_draw_scatter(df_2, "wind_speed", "active_power")
    # df_2.to_csv('./data/data2017.csv')
    # df_3.to_csv('./data/data2018_threeMonth.csv')

    # data_2017 = './data/data2017.csv'
    # X_tt, Y_tt = wt_preprocessing(data_2017, gs=True, rs=True, dt=True, ndt=True, ws=True, ap=True)

    # wt_params(X_tt, Y_tt)
    data_2018 = './data/data2018_threeMonth.csv'

    X_tt, Y_tt = wt_preprocessing(data_2018, gs=True, rs=True, dt=True, ndt=True, ws=True, ap=True)

    wt_params(X_tt, Y_tt)




if __name__ == '__main__':
    main()





'''
2017一整年的数据再次处理后可以提到60%多
远远不及一个月的效果,


'''