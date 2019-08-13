#coding:utf-8
'''
@Time: 2019/8/5 11:23
@author: Tokyo
@file: tk_24_trainingSet.py
@desc:
'''
from tk_18_data_preprocessing import wt_draw_scatter
import pandas as pd
df1 = pd.read_csv('./data/B/hfj162_test/96_1_2018-01-01~2018-07-01.csv', header=None)
df1.columns = ['date', 'Grid_ap', 'Grid_reap', 'state', 'voltage_phaseA', 'voltage_phaseB']
df2 = pd.read_csv('./data/B/hfj162_test/96_2_2018-01-01~2018-07-01.csv', header=None)
df2.columns = ['date', 'voltage_phaseC', 'Generation_active', 'Consumption_active', 'Generation_reactive', 'Consumption_reactive']
df3 = pd.read_csv('./data/B/hfj162_test/96_3_2018-01-01~2018-07-01.csv', header=None)
df3.columns = ['date', 'Generator_speed', 'Rotor_speed', 'Gearbox_oil_temp', 'Generator_bearing_tem_drive', 'Generator_bearing_tem_nondrive']
df4 = pd.read_csv('./data/B/hfj162_test/96_4_2018-01-01~2018-07-01.csv', header=None)
df4.columns = ['date', 'Power_factor', 'Grid_frequency', 'Pitch_angle', 'PCS_speed', 'PCS_torque']
df5 = pd.read_csv('./data/B/hfj162_test/96_5_2018-01-01~2018-07-01.csv', header=None)
df5.columns = ['date', 'PCS_rotor_temp', 'Nacelle_revolution', 'Wind_speed', 'Avg_Wind_speed_3s', 'Avg_wind_speed_5min']
df6 = pd.read_csv('./data/B/hfj162_test/96_6_2018-01-01~2018-07-01.csv', header=None)
df6.columns = ['date', 'Avg_wind_speed_30s', 'Avg_pitch_angle', 'Active_power', 'Avg_active_power_30s', 'Avg_active_pow_300s']
df9 = pd.read_csv('./data/B/hfj162_test/96_9_2018-01-01~2018-07-01.csv', header=None)
df9.columns = ['date', 'shutdown_time', 'bingwang_time', 'Gearbox_bearing_temp_A', 'Gearbox_bearing_temp_B', 'Gener_Stator_tempL1', 'Gener_Stator_tempL2', 'Gener_Stator_tempL3']
df_lst = [df2, df3, df4, df5, df6, df9]
df = df1
for i in range(len(df_lst)):
    df = pd.merge(df, df_lst[i], on='date')

df['date'] = pd.to_datetime(df['date'])
df['temp_1'] = df['Gearbox_oil_temp'].shift(5)
df['temp_2'] = df['Gearbox_oil_temp'].shift(10)
df = df.drop(['shutdown_time', 'bingwang_time'], axis=1)
df = df.drop_duplicates(subset=['date'])
print(len(df))
df.to_csv('./data/B/hfj162_test/raw_data162.csv', index=None)


# 合并5个训练集数据
# df1 = pd.read_csv('./data/B/train/hfj150_84/raw_data150.csv')
# df2 = pd.read_csv('./data/B/train/hfj152_86/raw_data152.csv')
# df3 = pd.read_csv('./data/B/train/hfj153_87/raw_data153.csv')
# df4 = pd.read_csv('./data/B/train/hfj156_90/raw_data156.csv')
# df5 = pd.read_csv('./data/B/train/hfj169_103/raw_data169.csv')
# df = pd.concat([df1, df2, df3, df4, df5])
# print(len(df))
# print(df.tail())
# df.to_csv('./data/B/train/tra_raw_data.csv', index=None)
# wt_draw_scatter(df, "Wind_speed", 'Active_power')

