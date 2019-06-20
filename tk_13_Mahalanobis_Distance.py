#coding:utf-8
'''
@Time: 2019/5/30 上午10:30
@author: Tokyo
@file: tk_13_Mahalanobis_Distance.py
@desc:使用马氏距离来进行异常检测
效果还不错,可以基本复现论文

思考:用马氏距离来进行数据预处理,目前聚类还是用的欧氏距离,前提是z-score标准化
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


    # 切分训练集,测试集
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.2, shuffle=False )


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
        xy_lst = [(X_train, Y_train), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = regr.predict(X_part)

            print(i)
            # 0--训练集, 1--测试集
            print(regr_name, "mean_squared_error", mean_squared_error(Y_part, Y_pred))
            print(regr_name, "mean_absolute_error", mean_absolute_error(Y_part, Y_pred))
            print(regr_name, "r2_score", r2_score(Y_part, Y_pred))



def WT_figure(features, label):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    import seaborn as sns
    import matplotlib.pyplot as plt
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values

    # 切分训练集,测试集
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.2, shuffle=False)


    xgb = XGBRegressor().fit(X_train, Y_train)
    Y_pred = xgb.predict(X_test)
    print(Y_pred)
    y_pre = pd.Series(Y_pred)
    y_pre.plot(c='g')
    # flatten 降维
    y_test = Y_test.flatten()
    y_test = pd.Series(y_test)
    y_test.plot(c='y')
    plt.ylim((50, 71))
    plt.xlim((0, 1000))
    plt.show()


def WT_MD(features, label):
    """
    基于马氏距离的异常检测
    :param features: 特征向量
    :param label: 待预测值
    """
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from scipy.spatial.distance import pdist, mahalanobis
    import seaborn as sns
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
    sns.distplot(MD, bins=65)
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
    print(X_test[445])
    pd.Series(MD_app).plot()
    plt.show()





def wt_params(X_train, Y_train):
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # cv_params = {'n_estimators': [75, 78, 70, 80]}
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight':[1, 2, 3, 4, 5, 6]}
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 70, 'max_depth': 4,
                    'min_child_weight': 6, 'subsample': 1, 'colsample_bytree': 1,
                    'gamma': 0.2, 'reg_alpha': 0.1, 'reg_lambda': 1}
    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2',
                                 cv=10, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evaluate_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evaluate_result))
    print('参数的最佳取值: {0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分: {0}'.format(optimized_GBM.best_score_))







def main():
    features, label = WT_preprocessing()
    # WT_modeling(features, label)
    # WT_figure(features, label)
    WT_MD(features, label)
    # wt_params(features, label)


if __name__ == '__main__':
    main()

'''
接下来的任务:
1.xgboost调参
2.增加数据,半年或一年

'''