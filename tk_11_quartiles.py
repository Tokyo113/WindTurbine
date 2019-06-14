
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
data: 5.23
author: Tokyo
desc:使用四分位法识别离群点,然后绘图显示
'''


data = pd.read_csv("./data/data_after_KMeans.csv")
data1 = data.drop("Unnamed: 0", axis=1)
print(len(data))


# 分桶
quartiles = pd.cut(data["wind_speed"], 20)


#
# def get_status(group):
#     q_interval = group.quantile(q=0.75) - group.quantile(q=0.25)
#     high = group.quantile(q=0.75) + 1.5*q_interval
#     low = group.quantile(q=0.25) - 1.5*q_interval
#
#     return {'q_high': high, 'q_low': low,
#             'count': group.count(), 'mean': group.mean()}
#
#
#
#
# grouped = data["active_power"].groupby(quartiles).apply(get_status)

# print(grouped.apply(get_status))
# print(type(grouped))



# 四分位法处理数据
s = []
for a, b in data["active_power"].groupby(quartiles):
    label = []
    q_interval = b.quantile(q=0.75) - b.quantile(q=0.25)
    high = b.quantile(q=0.75) + 1.5 * q_interval
    low = b.quantile(q=0.25) - 1.5 * q_interval
    for i in b:
        if (i < low) | (i > high):
            label.append(1)
        else:
            label.append(0)
    label_np = np.array(label)

    tk = pd.concat([b, pd.Series(label_np, index=b.index)], axis=1)
    s.append(tk)

s = pd.concat(s)
s = s.drop("active_power", axis=1)




data = pd.concat([data1, s], axis=1)
data.columns = list(data1.columns) + ["outlier"]
print(data.groupby("outlier").count())

# 绘制图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 三个类
normal = data[data["outlier"] == 0]
outlier = data[data["outlier"] == 1]

normal = normal.drop(["distance", "outlier", "category"], axis=1)

# 生成csv文件
# normal.to_csv('./data/data_preprocessing.csv', index=None)



# 正常点与离群点
plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
plt.scatter(outlier["wind_speed"], outlier["active_power"], c='r', s=3, alpha=.5)
plt.show()


