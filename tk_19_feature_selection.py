#coding:utf-8
'''
@Time: 2019/6/19 下午3:52
@author: Tokyo
@file: tk_19_feature_selection.py
@desc:
'''

import pandas as pd
import numpy as np
import seaborn as sns
from tk_18_data_preprocessing import wt_preprocessing
from tk_14_single_year import wt_params, WT_modeling
from tk_tools import wt_MD, wt_Cusum_change_point_detection, Pettitt_change_point_detection, Kendall_change_point_detection
from tk_13_Mahalanobis_Distance import WT_figure
def feature_selection(df):

    df = df.drop(['Avg_pitch_angle', 'Grid_frequency', 'Power_factor', 'Grid_ap', 'Grid_reap',
                  'Nacelle_revolution', 'Consumption_reactive', 'voltage_phaseA', 'voltage_phaseB',
                  'voltage_phaseB', 'Generation_reactive', 'Generator_bearing_tem_nondrive'], axis=1)

    df1 = df[['active_power', 'wind_speed', 'temp_1', 'temp_2', 'Gearbox_bearing_temp_A',
              'Gearbox_bearing_temp_A', 'Generator_speed', 'Rotor_speed', 'Gener_Stator_tempL3', 'voltage_phaseC',
              'PCS_torque', 'Consumption_active', 'Generator_bearing_tem_drive', 'Pitch_angle', 'PCS_rotor_temp',
              'date', 'state', 'Gearbox_oil_temp']]
    print(df1.head())
    return df1




def main():

    # 18年下半年 32715
    df = pd.read_csv('./data/data2018_half_year_train.csv')
    df1 = feature_selection(df)

    features_train, label_train, names_train = wt_preprocessing(df1, False)
    wt_params(features_train, label_train)

    # 测试集  5732
    # df_test = pd.read_csv('./data/data2018_April_test.csv')
    # df2 = feature_selection(df_test)

    # X_test, Y_test, name_test = wt_preprocessing(df2, False)
    # WT_modeling(features, label, X_test, Y_test)
    # md = wt_MD(features, label)
    # wt_Cusum_change_point_detection(md, 1000, 0.95)
    # print(Pettitt_change_point_detection(md))
    # print(Kendall_change_point_detection(md))


    # 一共38447条数据
    # df3 = pd.concat([df1, df2])
    # df3.to_csv('./data/data_train_test2018.csv', index = None)
    # features, label, names = wt_preprocessing(df3, False)

    # wt_params(features, label)
    # md = wt_MD(features, label)
    # WT_figure(features, label)


    # 变点检测
    # Cusum算法:变点[1198, 1923, 3713]
    # wt_Cusum_change_point_detection(md, 5000, 0.95)
    # Pettitt算法: 3772
    # print(Pettitt_change_point_detection(md))
    # # K算法:4266
    # print(Kendall_change_point_detection(md))






if __name__ == '__main__':
    main()