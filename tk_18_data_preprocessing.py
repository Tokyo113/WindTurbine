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
    # plt.xlim((0, 18))
    plt.xlabel('Wind Speed/(m/s)')
    plt.ylabel('Active Power/(kW)')
    plt.savefig('./data/figure/data_pre.png')
    plt.show()


def wt_preprocessing(data, method):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    data = data.drop("date", axis=1)
    data = data.drop("state", axis=1)

    # 得到特征和标签
    label = data["Gearbox_oil_temp"]
    features = data.drop("Gearbox_oil_temp", axis=1)
    names = features.columns.values
    if method:
        features = MinMaxScaler().fit_transform(features)
    else:
        features = StandardScaler().fit_transform(features)
    return features, label, names


def feature_RFE(features, label, names):
    """
    RFE包裹法选择特征
    :param data:
    输出特征的排名
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression


    lr = LinearRegression()

    rfe = RFE(lr, n_features_to_select=10)
    rfe.fit(features, label)
    print("Features sorted by their rank:")
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
    print(rfe.n_features_)


def feature_filter(features, label, names):
    """
    过滤法选择特征
    :param data:
    """
    from sklearn.feature_selection import SelectKBest, f_regression
    fil = SelectKBest(f_regression, k=10)
    fil.fit(features, label)
    print(sorted(zip(map(lambda x: round(x, 4), fil.scores_), names), reverse=True))


def feature_tree(features, label, names):
    """
    基于树的特征选择:--随机森林或极限树
    :param data:
    """
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    # clf = ExtraTreesRegressor(n_estimators=50)
    clf = RandomForestRegressor(n_estimators=20)
    clf.fit(features, label)
    print(features.shape)
    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(features)
    print(X_new.shape)







def main():

    # df = pd.read_csv('./data/hfj159/raw_93_2018.csv')
    #
    # # print(df.groupby("state").count())
    # # 切片:每隔5min取样
    # df = df.iloc[129550:697448:5]
    # print(df.tail())
    # df = df[df["Active_power"] > 1][df["state"] == 6]
    # df = df[df["Wind_speed"] <= 18]
    # df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
    # df = df.drop_duplicates(subset=['date'])
    # print(len(df))
    # df_1 = DBSCAN_cluster(df, 0.1, 25)
    # df_2 = Quartiles(df_1, 1.5, 50)
    #
    # # 2018年4月测试集
    # df_2.to_csv('./data/data2018_April_test.csv', index=None)


    # 数据预处理
    # df = pd.read_csv('./data/hfj159/raw_93_2018.csv')
    df = pd.read_csv('./data/final data/#159/data2018_Mar_test2.csv')
    df = df.iloc[49:6282:]
    df.to_csv('./data/final data/#159/data2018_April_test.csv', index=None)
    # 去掉功率为0的点
    # df = df[df["Active_power"] > 1][df["state"] == 6]
    # df = df[df["Active_power"] > 1]
    # df = df.drop_duplicates(subset=['date'])
    # # # print(df.groupby("state").count())
    # # # 切片:每隔5min取样
    # # print(df[df['date'] == '2018-03-01 00:00:00'])
    # # print(df[df['date'] == '2018-04-01 00:00:00'])
    # df = df.iloc[84939:129550:5]
    # print(df)
    # # #
    # # df = df[df["Wind_speed"] <= 20]
    # # df = df.drop_duplicates(subset=['date'])
    # df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
    # # #
    # wt_draw_scatter(df, 'wind_speed', 'active_power')
    # df_1 = DBSCAN_cluster(df, 0.1, 25)
    # df_2 = Quartiles(df_1, 2, 80)
    # # # df_2 = df_2.iloc[:6566]
    # # # # wt_draw_scatter(df_2, 'wind_speed', 'active_power')
    # df_2.to_csv('./data/final data/#159/data2018_Mar_test2.csv', index=None)
    #
    # print(df_2.describe())


    # 2018年下半年
    # df_2.to_csv('./data/data2018_half_year33.csv', index=None)

    # 33个特征,增加了前两个时刻温度
    # df_2.to_csv('./data/data2018_single_month_33.csv', index=None)


    # 特征选择
    data2018 = './data/data2018_half_year_train.csv'
    # data = pd.read_csv(data2018)
    # wt_draw_scatter(data, 'Gearbox_oil_temp', 'active_power')
    # features, label, names = wt_preprocessing(data, False)
    # feature_RFE(features, label, names)
    # feature_filter(features, label, names)
    # feature_tree(features, label, names)
    # wt_params(features, label)









if __name__ == '__main__':
    main()


'''
引入前两个时刻的温度值后效果很明显,
接下来:
尝试和对比模型融合方法: 线性回归+RF+XGB


选择出的特征:
RFE方法:
Avg_Wind_speed_3s, Avg_active_pow_300s, Avg_active_power_30s, Avg_wind_speed_30s, Avg_wind_speed_5min,Consumption_active
Consumption_reactive, Generation_reactive, Generation_reactive, Grid_frequency, PCS_rotor_temp, PCS_speed,PCS_torque
Rotor_speed,active_power, temp_1, temp_2, voltage_phaseA, voltage_phaseB, voltage_phaseC

过滤法:
temp_1, temp_2,Avg_wind_speed_30s,Avg_active_pow_300s,Avg_wind_speed_5min,Avg_active_power_30s
Avg_Wind_speed_3s,active_power, Grid_ap,PCS_speed,Rotor_speed, Generator_speed, wind_speed,
PCS_torque,Generator_bearing_tem_nondrive
 
 
 
目前暂定选择:
temp_1, temp_2, Avg_active_pow_300s, PCS_speed,Avg_wind_speed_30s,Generator_bearing_tem_nondrive,
active_power,Generator_speed,PCS_torque,Rotor_speed,Generator_bearing_tem_drive,PCS_rotor_temp,Avg_wind_speed_5min
Pitch_angle,Avg_active_power_30s,Generation_reactive,


'''