#coding:utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./data/alarm information/info_preprocessing.csv')

# 1.桨叶硬件故障
paddle = data[data["部件位置"] == "桨叶硬件"].drop("故障时长(分)", axis=1)
print(paddle.groupby("故障描述", sort=False).count())


# 2. 偏航故障
yawing = data[data["部件位置"] == "偏航"].drop("故障时长(分)", axis=1)
print(yawing["故障描述"].value_counts())
print(yawing.groupby("故障描述").count())
yawing = yawing.where((yawing["故障描述"] == "偏航误差过大") |
                      (yawing["故障描述"] == "右扭缆超限") |
                      (yawing["故障描述"] == "左扭缆超限") |
                      (yawing["故障描述"] == "机舱位置标定中")).fillna("其他")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
lbs = yawing["故障描述"].value_counts().index
explodes = [0.1 if i == "偏航误差过大" else 0 for i in lbs]
plt.pie(yawing["故障描述"].value_counts(normalize=True), explode=explodes, labels=lbs, autopct="%1.1f%%",
        colors=sns.color_palette("Reds"), pctdistance=0.7, shadow=True)
plt.title("偏航故障统计扇形图")
# plt.savefig("./data/yawing.png")
# plt.show()

# 3.控制器故障：
controller = data[data["部件位置"] == "控制器"].drop("故障时长(分)", axis=1)
print(controller.groupby("故障描述", sort=False).count())

