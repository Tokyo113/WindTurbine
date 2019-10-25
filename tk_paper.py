#coding:utf-8
'''
@Time: 2019/10/25 下午1:49
@author: Tokyo
@file: tk_paper.py
@desc:
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def data_preprocessing(filename):
    data = pd.read_csv(filename)

    # 添加前两时刻温度特征
    data['date'] = pd.to_datetime(data['date'])
    data['temp_1'] = data["Gearbox_oil_temp"].shift(5)
    data['temp_2'] = data["Gearbox_oil_temp"].shift(10)



    # 选择状态为6的数据并去掉state列
    data = data[data["state"] == 6].drop("state", axis=1)

    # 去掉功率为0的数据
    data = data[data["active_power"] > 1]

    # 取每10min数据
    data = data[11::5]
    df_oil = data["Gearbox_oil_temp"]
    # df_oil = data["Gearbox_oil_temperature"]
    df_bearing = data["Gearbox_bearing_temp_A"]


    # 处理前的散点图
    # plt.scatter(data["wind_speed"], data["active_power"], s=3, alpha=.5)
    # f = plt.figure(figsize=(12, 12))
    # ax1 = f.add_subplot(2, 1, 1)
    pd.Series(df_oil).plot(c='r', label = 'Gearbox oil temp')
    # ax2 = f.add_subplot(2, 1, 2)

    pd.Series(df_bearing).plot(c='y', label = 'Gearbox bearing temp')
    plt.legend(loc='lower left', ncol=2)
    plt.savefig('./data/figure/oil temp.png', dpi=400)
    plt.show()


    return data


if __name__ == '__main__':
    # 可以用
    data_preprocessing("./data/B/hfj149_83/test data.csv")
    # data_preprocessing("./data/C/test/hfj058_test_25/test data2.csv")