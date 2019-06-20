#coding:utf-8
'''
@Time: 2019/6/20 上午10:14
@author: Tokyo
@file: tk_tools.py
@desc:  常用的工具函数
'''
import pandas as pd
import numpy as np
import seaborn as sns


def wt_MD(features, label):
    """
    基于马氏距离的异常检测
    :param features: 特征向量
    :param label: 待预测值
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from scipy.spatial.distance import pdist, mahalanobis
    import matplotlib.pyplot as plt
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values



    # 切分训练集,测试集
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.2, shuffle=False)


    xgb = XGBRegressor().fit(X_train, Y_train)

    # 1.训练集,求得马氏距离检测异常的阈值
    Y_pred = xgb.predict(X_train)
    Y_train = Y_train.flatten()
    # 计算马氏距离
    err = Y_train - Y_pred

    X_ref = np.array([err, Y_train])
    X_ref = X_ref.T
    # 均值
    u = X_ref.mean(axis=0)
    delta = X_ref - u
    print(delta)
    print('******')

    cov = np.cov(X_ref.T)
    inv = np.linalg.inv(cov)
    print(cov)
    print(len(X_train))
    MD = []
    # MD_a = []
    for i in range(len(X_train)):
        md = np.dot(np.dot(delta[i], inv), delta[i].T)
        MD.append(np.sqrt(md))
        # 直接使用函数来计算
        # MD_a.append(mahalanobis(X_ref[i], u, inv))


    # 绘制概率密度直方图
    sns.distplot(MD, bins=45)
    plt.show()


    # 2.验证集(用于异常检测)
    Y_pred2 = xgb.predict(X_test)
    Y_test = Y_test.flatten()

    # 计算马氏距离
    err_test = Y_test - Y_pred2

    X_app = np.array([err_test, Y_test])
    X_app = X_app.T
    # 均值u和协方差矩阵inv均使用训练集得出的值
    MD_app = []
    for i in range(len(X_test)):
        MD_app.append(mahalanobis(X_app[i], u, inv))

    md = pd.Series(MD_app)
    md[0:200].plot()
    # 滑动窗口法 
    md[0:200].rolling(6).mean().plot()

    plt.show()
