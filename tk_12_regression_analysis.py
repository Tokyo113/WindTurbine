#coding:utf-8
'''
@Time: 2019/5/24 上午11:49
@author: Tokyo
@file: tk_12_regression_analysis.py
@desc: 建立回归模型:线性回归,决策树,SVM,随机森林,AdaBoost,GBDT,XgBoost
        而且做出图像对比预测值和实际值
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def WT_preprocessing(gs=False, rs=False, dt=False, ndt=False, ws=False, ap=False):

    # 1.读入数据
    df = pd.read_csv("./data/data_preprocessing.csv")
    # 得到标签(待预测值)
    label = df["Gearbox_oil_temperature"]

    df = df.drop("Gearbox_oil_temperature", axis=1)



    # 特征处理
    scaler_lst = [gs, rs, dt, ndt, ws, ap]
    column_lst = ["Generator_speed", "Rotor_speed",
                  "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
                  "wind_speed", "wind_speed"]

    for i in range(len(scaler_lst)):
        if scaler_lst[i]:
            df[column_lst[i]] = \
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

        else:
            df[column_lst[i]] = \
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

    return df, label


def WT_modeling(features, label):
    from sklearn.model_selection import train_test_split
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values

    # 切分训练集,测试集,验证集
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    models = []
    # 线性回归
    # models.append(("LinearRegression", LinearRegression()))
    models.append(("Ridge", Ridge(alpha=0.6)))
    # models.append(("Lasso", Lasso(alpha=0.002)))
    # models.append(("Logistic", LogisticRegression()))
    # 决策树回归
    models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    # 支持向量回归(误差很大)
    # models.append(("SVR", SVR(C=100000)))
    models.append(("RandomForestRegressor", RandomForestRegressor()))
    # AdaBoostRegressor  base_estimator=DecisionTreeRegressor默认
    models.append(("AdaBoostRegressor", AdaBoostRegressor()))
    # GBDT 回归
    models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    models.append(("XGBoost", XGBRegressor()))



    for regr_name, regr in models:
        regr.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = regr.predict(X_part)

            print(i)
            # 0--训练集, 1--验证集, 2--测试集
            print(regr_name, "mean_squared_error", mean_squared_error(Y_part, Y_pred))
            print(regr_name, "mean_absolute_error", mean_absolute_error(Y_part, Y_pred))
            print(regr_name, "median_absolute_error", median_absolute_error(Y_part, Y_pred))



def WT_figure(features, label):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    import seaborn as sns
    import matplotlib.pyplot as plt
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values

    # 切分训练集,测试集,验证集
    X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)

    xgb = XGBRegressor().fit(X_train, Y_train)
    Y_pred = xgb.predict(X_test)
    print(Y_pred)
    y_pre = pd.Series(Y_pred)
    y_pre[150:250].plot(c='g')
    # flatten 降维
    y_test = Y_test.flatten()
    y_test = pd.Series(y_test)
    y_test[150:250].plot(c='y')
    plt.show()




def main():
    features, label = WT_preprocessing()
    WT_modeling(features, label)
    # WT_figure(features, label)


if __name__ == '__main__':
    main()