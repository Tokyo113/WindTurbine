#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/raw_data/raw_data1.csv").head(45000)
# 去掉功率为0的点
data = data[data["active_power"] > 1]
print(data.describe())
# 切片:每隔5min取样
data1 = data.iloc[0:35592:5]
print(data1.head())