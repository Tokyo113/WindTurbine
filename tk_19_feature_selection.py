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
from tk_14_single_year import wt_params

def feature_selection(df):

    df = df.drop(['Avg_pitch_angle', 'Grid_frequency', 'Power_factor', 'Grid_ap',
                  'Nacelle_revolution', 'voltage_phaseA', 'voltage_phaseB'], axis=1)
    print(df.describe())
    return df




def main():
    df = pd.read_csv('./data/data2018_single_month_33.csv')
    df1 = feature_selection(df)
    features, label, names = wt_preprocessing(df1, False)
    wt_params(features, label)

    # 测试集
    df_test = pd.read_csv('./data/data2018_single_month_test.csv')
    df2 = feature_selection(df_test)
    X_test, Y_test, name_test = wt_preprocessing(df2, False)
    


if __name__ == '__main__':
    main()