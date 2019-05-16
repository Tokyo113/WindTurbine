#coding:utf-8
import pandas as pd
import numpy as np


data1 = pd.read_csv('./data/raw_data/3.csv', header=None)
data2 = pd.read_csv('./data/raw_data/5.csv', header=None)
data3 = pd.read_csv('./data/raw_data/6.csv', header=None)
# 引入风机当前状态
data4 = pd.read_csv('./data/raw_data/1.csv', header=None)

data1[6] = data2[3]
data1[7] = data3[3]
data1[8] = data4[3]
data1.columns = ["date", "Generator_speed", "Rotor_speed", "Gearbox_oil_temperature",
                 "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
                 "wind_speed", "active_power", "state"]
print(data1.groupby("state").count())

# data1.to_csv("./data/raw_data/raw_data_state.csv", index=None)


