#coding:utf-8
import numpy as np
import pandas as pd

# 上半年
df_2018_1a = pd.read_csv("./data/2018/107号风机第一组5个数据2018-01-01~2018-07-01.csv", header=None)
df_2018_3a = pd.read_csv("./data/2018/107号风机第三组5个数据2018-01-01~2018-07-01.csv", header=None)
df_2018_5a = pd.read_csv("./data/2018/107号风机第五组5个数据2018-01-01~2018-07-01.csv", header=None)
df_2018_6a = pd.read_csv("./data/2018/107号风机第六组5个数据2018-01-01~2018-07-01.csv", header=None)

df_2018_3a[6] = df_2018_5a[3]
df_2018_3a[7] = df_2018_6a[3]
df_2018_3a[8] = df_2018_1a[3]

df_2018_3a.columns = ["date", "Generator_speed", "Rotor_speed", "Gearbox_oil_tem",
                     "Generator_bearing_tem_drive", "Generator_bearing_tem_nondrive",
                     "wind_speed", "active_power", "state"]
print(len(df_2018_3a))

# 下半年
df_2018_1b = pd.read_csv("./data/2018/107号风机第一组5个数据2018-07-01~2019-01-01.csv", header=None)
df_2018_3b = pd.read_csv("./data/2018/107号风机第三组5个数据2018-07-01~2019-01-01.csv", header=None)
df_2018_5b = pd.read_csv("./data/2018/107号风机第五组5个数据2018-07-01~2019-01-01.csv", header=None)
df_2018_6b = pd.read_csv("./data/2018/107号风机第六组5个数据2018-07-01~2019-01-01.csv", header=None)

df_2018_3b[6] = df_2018_5b[3]
df_2018_3b[7] = df_2018_6b[3]
df_2018_3b[8] = df_2018_1b[3]

df_2018_3b.columns = ["date", "Generator_speed", "Rotor_speed", "Gearbox_oil_tem",
                     "Generator_bearing_tem_drive", "Generator_bearing_tem_nondrive",
                     "wind_speed", "active_power", "state"]
print(len(df_2018_3b))
data_2018 = df_2018_3a.append(df_2018_3b)
data_2018.to_csv('./data/2018/raw_data2018.csv', index=None)