#coding:utf-8

'''
@author: Tokyo
@desc:
'''
import pandas as pd
import numpy as np
import seaborn as sns
# # 2018下半年
# df1 = pd.read_csv('./data/173/1_2018-07-01~2019-01-01.csv', header=None)
# df1.columns = ['date', 'Grid_ap', 'Grid_reap', 'state', 'voltage_phaseA', 'voltage_phaseB']
# df2 = pd.read_csv('./data/173/2_2018-07-01~2019-01-01.csv', header=None)
# df2.columns = ['date', 'voltage_phaseC', 'Generation_active', 'Consumption_active', 'Generation_reactive', 'Consumption_reactive']
# df3 = pd.read_csv('./data/173/3_2018-07-01~2019-01-01.csv', header=None)
# df3.columns = ['date', 'Generator_speed', 'Rotor_speed', 'Gearbox_oil_temp', 'Generator_bearing_tem_drive', 'Generator_bearing_tem_nondrive']
# df4 = pd.read_csv('./data/173/4_2018-07-01~2019-01-01.csv', header=None)
# df4.columns = ['date', 'Power_factor', 'Grid_frequency', 'Pitch_angle', 'PCS_speed', 'PCS_torque']
# df5 = pd.read_csv('./data/173/5_2018-07-01~2019-01-01.csv', header=None)
# df5.columns = ['date', 'PCS_rotor_temp', 'Nacelle_revolution', 'Wind_speed', 'Avg_Wind_speed_3s', 'Avg_wind_speed_5min']
# df6 = pd.read_csv('./data/173/6_2018-07-01~2019-01-01.csv', header=None)
# df6.columns = ['date', 'Avg_wind_speed_30s', 'Avg_pitch_angle', 'Active_power', 'Avg_active_power_30s', 'Avg_active_pow_300s']
# df9 = pd.read_csv('./data/173/9_2018-07-01~2019-01-01.csv', header=None)
# df9.columns = ['date', 'shutdown_time', 'bingwang_time', 'Gearbox_bearing_temp_A', 'Gearbox_bearing_temp_B', 'Gener_Stator_tempL1', 'Gener_Stator_tempL2', 'Gener_Stator_tempL3']

# 24号风机 2017下半年，2018上半年
# df1 = pd.read_csv('./data/hfj057/24_1_2018-01-01~2018-07-01.csv', header=None)
# df1.columns = ['date', 'Grid_ap', 'Grid_reap', 'state', 'voltage_phaseA', 'voltage_phaseB']
# df2 = pd.read_csv('./data/hfj057/24_2_2018-01-01~2018-07-01.csv', header=None)
# df2.columns = ['date', 'voltage_phaseC', 'Generation_active', 'Consumption_active', 'Generation_reactive', 'Consumption_reactive']
# df3 = pd.read_csv('./data/hfj057/24_3_2018-01-01~2018-07-01.csv', header=None)
# df3.columns = ['date', 'Generator_speed', 'Rotor_speed', 'Gearbox_oil_temp', 'Generator_bearing_tem_drive', 'Generator_bearing_tem_nondrive']
# df4 = pd.read_csv('./data/hfj057/24_4_2018-01-01~2018-07-01.csv', header=None)
# df4.columns = ['date', 'Power_factor', 'Grid_frequency', 'Pitch_angle', 'PCS_speed', 'PCS_torque']
# df5 = pd.read_csv('./data/hfj057/24_5_2018-01-01~2018-07-01.csv', header=None)
# df5.columns = ['date', 'PCS_rotor_temp', 'Nacelle_revolution', 'Wind_speed', 'Avg_Wind_speed_3s', 'Avg_wind_speed_5min']
# df6 = pd.read_csv('./data/hfj057/24_6_2018-01-01~2018-07-01.csv', header=None)
# df6.columns = ['date', 'Avg_wind_speed_30s', 'Avg_pitch_angle', 'Active_power', 'Avg_active_power_30s', 'Avg_active_pow_300s']
# df9 = pd.read_csv('./data/hfj057/24_9_2018-01-01~2018-07-01.csv', header=None)
# df9.columns = ['date', 'shutdown_time', 'bingwang_time', 'Gearbox_bearing_temp_A', 'Gearbox_bearing_temp_B', 'Gener_Stator_tempL1', 'Gener_Stator_tempL2', 'Gener_Stator_tempL3']

# 93号风机 HFJ159
df1 = pd.read_csv('./data/hfj159/93_1_2017-07-01~2018-01-01.csv', header=None)
df1.columns = ['date', 'Grid_ap', 'Grid_reap', 'state', 'voltage_phaseA', 'voltage_phaseB']
df2 = pd.read_csv('./data/hfj159/93_2_2017-07-01~2018-01-01.csv', header=None)
df2.columns = ['date', 'voltage_phaseC', 'Generation_active', 'Consumption_active', 'Generation_reactive', 'Consumption_reactive']
df3 = pd.read_csv('./data/hfj159/93_3_2017-07-01~2018-01-01.csv', header=None)
df3.columns = ['date', 'Generator_speed', 'Rotor_speed', 'Gearbox_oil_temp', 'Generator_bearing_tem_drive', 'Generator_bearing_tem_nondrive']
df4 = pd.read_csv('./data/hfj159/93_4_2017-07-01~2018-01-01.csv', header=None)
df4.columns = ['date', 'Power_factor', 'Grid_frequency', 'Pitch_angle', 'PCS_speed', 'PCS_torque']
df5 = pd.read_csv('./data/hfj159/93_5_2017-07-01~2018-01-01.csv', header=None)
df5.columns = ['date', 'PCS_rotor_temp', 'Nacelle_revolution', 'Wind_speed', 'Avg_Wind_speed_3s', 'Avg_wind_speed_5min']
df6 = pd.read_csv('./data/hfj159/93_6_2017-07-01~2018-01-01.csv', header=None)
df6.columns = ['date', 'Avg_wind_speed_30s', 'Avg_pitch_angle', 'Active_power', 'Avg_active_power_30s', 'Avg_active_pow_300s']
df7 = pd.read_csv('./data/hfj159/93_7_2017-07-01~2018-01-01.csv', header=None)
df7.columns = ['date', 'Avg_generator_speed_3s', 'Yawing_error', 'Nacelle revolution_B', 'Wind_direction_3s', 'Wind_direction_5min']
df9 = pd.read_csv('./data/hfj159/93_9_2017-07-01~2018-01-01.csv', header=None)
df9.columns = ['date', 'shutdown_time', 'bingwang_time', 'Gearbox_bearing_temp_A', 'Gearbox_bearing_temp_B', 'Gener_Stator_tempL1', 'Gener_Stator_tempL2', 'Gener_Stator_tempL3']

df_lst = [df2, df3, df4, df5, df6, df7, df9]
df = df1
for i in range(len(df_lst)):
    df = pd.merge(df, df_lst[i], on='date')
    
    
df['date'] = pd.to_datetime(df['date'])
print(df['date'].dtype)
df['temp_1'] = df['Gearbox_oil_temp'].shift(5)
df['temp_2'] = df['Gearbox_oil_temp'].shift(10)
df = df.drop(['shutdown_time', 'bingwang_time'], axis=1)
print(df.head())
print(len(df))

# hfj057 号风机--24号原始数据
# df.to_csv('./data/hfj057/raw_24_2017.csv', index=None)

# hfj159 号风机--93号原始数据
# df.to_csv('./data/hfj159/raw_93_2018.csv', index=None)
# 带第七组数据
df.to_csv('./data/hfj159/raw_93_2018_2.csv', index=None)
# df.to_csv('./data/hfj057/raw_24_2018.csv', index=None)
