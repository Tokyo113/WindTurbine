#coding:utf-8
'''
@Time: 2019/6/10 下午5:27
@author: Tokyo
@file: tk_15_three_months.py
@desc: 18年前三个月建立正常模型,第四个月测试
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def wt_preprocessing(filename, gs=False, rs=False, dt=False, ndt=False, ws=False, ap=False):
    """
    对各列特征进行标准化或者归一化处理
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # 1.读入数据
    df = pd.read_csv(filename)
    df = df.drop("date", axis=1)
    # 得到标签(待预测值)
    label = df["Gearbox_oil_tem"]

    df = df.drop("Gearbox_oil_tem", axis=1)

    # 特征处理
    scaler_lst = [gs, rs, dt, ndt, ws, ap]
    column_lst = ["Generator_speed", "Rotor_speed",
                  "Generator_bearing_tem_drive", "Generator_bearing_tem_nondrive",
                  "wind_speed", "active_power"]

    for i in range(len(scaler_lst)):
        if scaler_lst[i]:
            df[column_lst[i]] = \
                MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

        else:
            df[column_lst[i]] = \
                StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]

    return df, label


def wt_modeling(X_tt, Y_tt, X_test, Y_test):
    from sklearn.model_selection import train_test_split

    X_tt = pd.DataFrame(X_tt).values
    Y_tt = pd.DataFrame(Y_tt).values
    # 2017年数据分出一部分作为验证集
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_tt, Y_tt, test_size=0.2, shuffle=None)

    X_test = pd.DataFrame(X_test).values
    Y_test = pd.DataFrame(Y_test).values



    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    models = []
    # 线性回归
    # models.append(("LinearRegression", LinearRegression()))
    # models.append(("Ridge", Ridge(alpha=0.6)))
    # models.append(("Lasso", Lasso(alpha=0.002)))
    # 决策树回归
    # models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    # 支持向量回归(误差很大)
    # models.append(("SVR", SVR(C=100000)))
    models.append(("RandomForestRegressor", RandomForestRegressor()))
    # AdaBoostRegressor  base_estimator=DecisionTreeRegressor默认
    models.append(("AdaBoostRegressor", AdaBoostRegressor()))
    # GBDT 回归
    models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    models.append(("XGBoost", XGBRegressor(max_depth=10, n_estimators=8000, learning_rate=0.1)))

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
            print(regr_name, "r2_score", r2_score(Y_part, Y_pred))


def wt_params(X_train, Y_train):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    cv_params = {'n_estimators': [1000, 2000, 3000, 4000]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5,
                    'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 1,
                    'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2',
                                 cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evaluate_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evaluate_result))
    print('参数的最佳取值: {0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分: {0}'.format(optimized_GBM.best_score_))


def main():
    filename = "./data/year/data_pre2017.csv"
    df, label = wt_preprocessing(filename)

    dd = pd.read_csv(filename)
    dd = dd[8786:11539]
    X_tt = df[0:8786]
    Y_tt = label[0:8786]
    X_test = df[8786:11539]
    Y_test = label[8786:11539]
    # wt_modeling(X_tt, Y_tt, X_test, Y_test)
    wt_params(X_tt, Y_tt)





if __name__ == '__main__':
    main()



'''
初始参数
def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
06.11
交叉验证显示结果并不好======>>特征远远不够,继续增加特征!!!

'''