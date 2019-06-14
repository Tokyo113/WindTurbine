#coding:utf-8

'''
@author: Tokyo
@desc:
'''
import pandas as pd
import numpy as np
import seaborn as sns

df1 = pd.read_csv('./data/173/1_2018-07-01~2019-01-01.csv', header=None)
df1.columns = ['date', 'Grid_ap', 'Grid_reap', 'state', 'voltage_phaseA', 'voltage_phaseB']
df2 = pd.read_csv('./data/173/2_2018-07-01~2019-01-01.csv', header=None)
df2.columns = ['date', 'voltage_phaseC', 'Generation_active', 'Consumption_active', 'Generation_reactive', 'Consumption_reactive']
df3 = pd.read_csv('./data/173/3_2018-07-01~2019-01-01.csv', header=None)
df3.columns = ['date', 'Generator_speed', 'Rotor_speed', 'Gearbox_oil_temp', 'Generator_bearing_tem_drive', 'Generator_bearing_tem_nondrive']
df4 = pd.read_csv('./data/173/4_2018-07-01~2019-01-01.csv', header=None)
df4.columns = ['date', 'Power_factor', 'Grid_frequency', 'Pitch_angle', 'PCS_speed', 'PCS_torque']
df5 = pd.read_csv('./data/173/5_2018-07-01~2019-01-01.csv', header=None)
df5.columns = ['date', 'PCS_rotor_temp', 'Nacelle_revolution', 'Wind_speed', 'Avg_Wind_speed_3s', 'Avg_wind_speed_5min']
df6 = pd.read_csv('./data/173/6_2018-07-01~2019-01-01.csv', header=None)
df6.columns = ['date', 'Avg_wind_speed_30s', 'Avg_pitch_angle', 'Active_power', 'Avg_active_power_30s', 'Avg_active_pow_300s']


df_lst = [df2, df3, df4, df5, df6]
df = df1
for i in range(len(df_lst)):
    df = pd.merge(df, df_lst[i], on='date')

print(df.head())
df.to_csv('./data/year/feature2018_31.csv', index=None)