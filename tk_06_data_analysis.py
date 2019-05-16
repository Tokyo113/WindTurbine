#coding:utf-8
import pandas as pd
import numpy as np

data = pd.read_csv("./data/raw_data.csv")
# 去掉功率为0的点
data = data[data["active_power"] > 1]
print(data.describe())