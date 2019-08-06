#coding:utf-8
'''
@Time: 2019/8/5 下午3:06
@author: Tokyo
@file: tk_25_preprocessing.py
@desc:
'''

import pandas as pd
from tk_18_data_preprocessing import wt_draw_scatter
from tk_14_single_year import Quartiles, DBSCAN_cluster



# type C

# 训练集数据,5个月
# df = pd.read_csv('./data/C/train/tra_raw_data.csv')
# # 切片:每隔5min取样
# df = df.iloc[::5]
# df = df[df["Active_power"] > 1][df["state"] == 6]
# # df = df[df["Wind_speed"] <= 18]
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# print(len(df))
# df_1 = DBSCAN_cluster(df, 0.1, 70)
# df_2 = Quartiles(df_1, 2, 200)
# wt_draw_scatter(df_2, "wind_speed", "active_power")
# # 2018年4月测试集
# df_2.to_csv('./data/C/train/training set.csv', index=None)

# 正常模型验证数据, hfj061
# 一个月数据为测试数据,2018/07/01~2018/07/30
# df = pd.read_csv('./data/C/hfj061_28/raw_data061.csv')
# df = df[df["Active_power"] > 1][df["state"] == 6]
# df = df.iloc[::5]
# df = df.iloc[0:6864]
# # print(df[df['date'] == '2018-07-01 00:00:00'])
# # print(df[df['date'] == '2018-08-01 00:00:00'])
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# print(df)
# df_1 = DBSCAN_cluster(df, 0.1, 40)
# df_2 = Quartiles(df_1, 1.5, 200)
# # wt_draw_scatter(df_2, "wind_speed", "active_power")
# # # 2018年4月测试集
# df_2.to_csv('./data/C/hfj061_28/validation set.csv', index=None)

# 故障测试数据
# hfj038 03/01~04/05
df = pd.read_csv('./data/C/test/hfj038_test_5/raw_data038.csv')

df = df.iloc[84939:135299:5]
df = df[df["Active_power"] > 1][df["state"] == 6]
# print(df[df['date'] == '2018-03-01 00:00:00'])
# print(df[df['date'] == '2018-04-05 00:00:00'])
df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
print(len(df))
df_1 = DBSCAN_cluster(df, 0.1, 10)
df_2 = Quartiles(df_1, 1.5, 200)
# wt_draw_scatter(df_2, "wind_speed", "active_power")
# # # 2018年3月测试集
df_2.to_csv('./data/C/test/hfj038_test_5/test data1.csv', index=None)

# hfj058 04/01~04/30
# df = pd.read_csv('./data/C/test/hfj058_test_25/raw_data058.csv')

# # df = df.iloc[129550:172603:5]
# df = df.iloc[::5]
# # print(df[df['date'] == '2018-04-01 00:00:00'])
# # print(df[df['date'] == '2018-05-01 00:00:00'])
# df = df[df["Active_power"] > 1][df["state"] == 6]
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# # print(len(df))
# df_1 = DBSCAN_cluster(df, 0.1, 15)
# df_2 = Quartiles(df_1, 1.5, 200)
# # # wt_draw_scatter(df_2, "wind_speed", "active_power")
# 2018年4月测试集
# df_2.to_csv('./data/C/test/hfj058_test_25/test data2.csv', index=None)


# type B
# 训练集数据 5个月
# df = pd.read_csv('./data/B/train/tra_raw_data.csv')
# # # 切片:每隔5min取样
# df = df.iloc[::5]
# df = df[df["Active_power"] > 1][df["state"] == 6]
#
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# print(len(df))
# df_1 = DBSCAN_cluster(df, 0.1, 100)
# df_2 = Quartiles(df_1, 2, 200)
# # wt_draw_scatter(df_2, "wind_speed", "active_power")
# # # 2018年4月测试集
# df_2.to_csv('./data/B/train/training set.csv', index=None)


# 验证集数据 hfj149
# df = pd.read_csv('./data/B/hfj149_83/raw_data149.csv')
# #
# df = df.iloc[219259:262252:5]

# print(df[df['date'] == '2018-12-01 00:00:00'])
# print(df[df['date'] == '2018-12-31 00:00:00'])
# df = df[df["Active_power"] > 1][df["state"] == 6]
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# print(len(df))
# df_1 = DBSCAN_cluster(df, 0.1, 10)
# df_2 = Quartiles(df_1, 1.5, 200)
# # # # wt_draw_scatter(df_2, "wind_speed", "active_power")
# # 2018年4月测试集
# df_2.to_csv('./data/B/hfj149_83/test data.csv', index=None)


# 论文图片:
# df = pd.read_csv('./data/raw_93_2018.csv')
#
# df = df.iloc[::5]
# df = df.drop_duplicates(subset=['date'])
# # print(df[df['date'] == '2018-04-01 00:00:00'])
# # print(df[df['date'] == '2018-05-01 00:00:00'])
# df = df[df["Active_power"] > 1][df["state"] == 6]
# df = df[df["Wind_speed"] <= 21]
# df.rename(columns={'Active_power': 'active_power', 'Wind_speed': 'wind_speed'}, inplace=True)
# # print(len(df))
# df_1 = DBSCAN_cluster(df, 0.1, 70)
# df_2 = Quartiles(df_1, 1.5, 200)
# wt_draw_scatter(df_2, "wind_speed", "active_power")
# 2018年4月测试集
# df_2.to_csv('./data/C/test/hfj058_test_25/test data2.csv', index=None)
