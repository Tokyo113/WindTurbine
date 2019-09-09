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
    import matplotlib.patches
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.scatter(data[x], data[y], c='g', s=3, alpha=0.5)
    # plt.xlim((0, 18))
    plt.xlabel('Wind Speed/(m/s)')
    plt.ylabel('Active Power/(kW)')
    cir1 = matplotlib.patches.Ellipse((13.5, 1120), 8, 150, facecolor='None', edgecolor='red', lw=2, alpha=1)
    cir2 = matplotlib.patches.Ellipse((16, 1530), 10, 520, facecolor='None', edgecolor='red', lw=2, alpha=1)
    cir3 = matplotlib.patches.Ellipse((13.5, 0), 10, 150, facecolor='None', edgecolor='red', lw=2, alpha=1)
    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.add_patch(cir3)
    plt.annotate('Type 1', xy=(12, 75), xycoords='data', fontsize=10,
                 xytext=(+60, +10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Type 2', xy=(13, 1030), xycoords='data', fontsize=10,
                 xytext=(+70, -50), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Type 3', xy=(15, 1800), xycoords='data', fontsize=10,
                 xytext=(+60, +10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.savefig('./data/figure/raw data.png', dpi=400)
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
    # data_2018 = './data/data2018_threeMonth.csv'
    #
    # X_tt, Y_tt = wt_preprocessing(data_2018, gs=True, rs=True, dt=True, ndt=True, ws=True, ap=True)
    #
    # wt_params(X_tt, Y_tt)
    df = pd.read_csv('./data/B/train/hfj156_90/raw_data156.csv')
    # df = pd.read_csv('./data/B//hfj149_83/test data.csv')
    df = df[0:200000]
    wt_draw_scatter(df, "Wind_speed", "Active_power")




if __name__ == '__main__':
    main()





'''
2017一整年的数据再次处理后可以提到60%多
远远不及一个月的效果,


'''