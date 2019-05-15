import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/alarm information/info_preprocessing.csv")
df = data.drop("故障描述", axis=1)
print(df.describe())
# 各个部件的平均停机时间
print(df.groupby("部件位置").mean())

# 1.齿轮箱故障详细分析
gearBox = data[data["部件位置"] == "齿轮箱"].drop("故障时长(分)", axis=1)
print(gearBox.groupby("故障描述").count())


# 柱状图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("故障描述")
plt.xlabel("number")
plt.ylabel("故障描述")
# plt.xticks(np.arange(len(gearBox["故障描述"].value_counts()))+0.5, gearBox["故障描述"].value_counts().index)
plt.yticks(np.arange(len(gearBox["故障描述"].value_counts()))+0.5, gearBox["故障描述"].value_counts().index)
# 设置显示范围  加0.5向右平移
plt.axis([-20, 3000, 0, 15])
plt.barh(np.arange(len(gearBox["故障描述"].value_counts()))+0.5, gearBox["故障描述"].value_counts())
# 标注具体数字
for y, x in zip(np.arange(len(gearBox["故障描述"].value_counts()))+0.5, gearBox["故障描述"].value_counts()):
    # x, y 标注的值是y，水平，竖直位置
    plt.text(x, y, x, ha="left", va="center")
# plt.savefig("./data/tk1.png")
plt.show()
