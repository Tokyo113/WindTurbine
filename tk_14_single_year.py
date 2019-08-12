#coding:utf-8
'''
@Time: 2019/5/31 下午2:06
@author: Tokyo
@file: tk_14_single_year.py
@desc:取一整年的数据进行分析和预处理
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def data_preprocessing(filename):
    data = pd.read_csv(filename)

    # 添加前两时刻温度特征
    data['date'] = pd.to_datetime(data['date'])
    data['temp_1'] = data['Gearbox_oil_tem'].shift(10)
    data['temp_2'] = data['Gearbox_oil_tem'].shift(20)



    # 选择状态为6的数据并去掉state列
    data = data[data["state"] == 6].drop("state", axis=1)

    # 去掉功率为0的数据
    data = data[data["active_power"] > 1]

    # 取每10min数据
    data = data[21::10]

    print(data.describe())

    # 处理前的散点图
    # plt.scatter(data["wind_speed"], data["active_power"], s=3, alpha=.5)
    # plt.show()
    print(data.mean())
    return data


def K_Means(data):
    """
    后续考虑问题:
    1.如何选择聚类个数?
    2.如何确定阈值?
    3.聚类标准:目前是欧氏距离,用马氏距离?
    :param data: 数据集
    """
    from sklearn.cluster import KMeans
    # 参数初始化
    # 聚类个数
    k = 15
    # 离散点阈值   (3, 1.97)  (15, 1.95)
    threshold = 1.95
    # 聚类最大循环次数
    iteration = 500
    # 数据标准化  z-score
    # 马氏距离? 博客标准化的缺陷
    data1 = data.drop("date", axis=1)
    data_zs = 1.0 * (data1 - data1.mean()) / data1.std()

    # 只使用风速和功率进行聚类
    data_tk = data_zs[["wind_speed", "active_power"]]

    model = KMeans(n_clusters=k, max_iter=iteration, )
    model.fit(data_tk)

    # 添加类别属性列
    cluster_data = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    cluster_data.columns = list(data.columns) + ["category"]
    # print(cluster_data.groupby("category").count())

    norm = []
    for i in range(k):
        # norm_tmp = data_zs[["Generator_speed", "Rotor_speed", "Gearbox_oil_temperature",
        #              "Generator_bearing_temperature_drive", "Generator_bearing_temperature_nondrive",
        #              "wind_speed", "active_power"]][cluster_data["category"] == i]-model.cluster_centers_[i]
        # 只考虑风速和功率计算距离
        norm_tmp = data_zs[["wind_speed", "active_power"]][cluster_data["category"] == i] - model.cluster_centers_[i]
        # 求绝对距离
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)

        # 求相对距离
        norm.append(norm_tmp / norm_tmp.median())

    norm = pd.concat(norm)
    cluster_data = pd.concat([cluster_data, norm], axis=1)
    cluster_data.columns = list(data.columns) + ["category"] + ["distance"]
    return cluster_data


def DBSCAN_cluster(data, eps, minPts):
    """
    效果还行,收敛时间较长
    基本可以识别离群点
    :param data:
    """
    from sklearn.cluster import DBSCAN
    # (0.2, 250)
    eps = eps  # 0.2
    minPts = minPts  # 250
    # 数据标准化  z-score
    # 马氏距离? 博客标准化的缺陷
    data1 = data.drop("date", axis=1)
    data_zs = 1.0 * (data1 - data1.mean()) / data1.std()

    # 只使用风速和功率进行聚类
    data_tk = data_zs[["wind_speed", "active_power"]]
    model = DBSCAN(eps=eps, min_samples=minPts, algorithm='kd_tree')
    model.fit(data_tk)

    # 添加类别属性列
    cluster_data = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    cluster_data.columns = list(data.columns) + ["category"]
    print(cluster_data.groupby("category").count())

    # 防止将上部曲线识别为异常点  两种风机:1500---1550和2000
    cluster_data["category"][(cluster_data["category"] == -1) & (cluster_data["active_power"] > 1500)] = 0
    outier = cluster_data[(cluster_data["category"] == -1)]
    normal = cluster_data[cluster_data["category"] != -1]
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 正常点与离群点
    plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5, label='normal points')
    plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5, label='outliers')
    plt.xlabel('Wind Speed/(m/s)')
    plt.ylabel('Active Power/(kW)')
    plt.legend(loc='upper left')
    # plt.savefig('./data/paper/outlier1.png')
    plt.show()


    normal = normal.drop("category", axis=1)
    print(normal.describe())
    return normal


def draw_clusters(cluster_data):
    """
    K_Means聚类后作图
    :param cluster_data:
    """
    threshold = 2.5
    outier = cluster_data[cluster_data["distance"] >= threshold]
    normal = cluster_data[cluster_data["distance"] < threshold]
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 正常点与离群点
    plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5)
    plt.scatter(outier["wind_speed"], outier["active_power"], c='r', s=3, alpha=.5)
    plt.show()


def Quartiles(data, k, cut_num):
    """
    四分位法处理数据
    :param data:
    :param k: k=1.5~3
    :param cut_num: 分桶数
    :return: 去掉离群数据后的数据集
    """
    data1 = data.drop("date", axis=1)
    column_name = list(data.columns)
    # 分桶
    quartiles = pd.cut(data["wind_speed"], cut_num)

    # 四分位法处理数据
    s = []

    for a, b in data["active_power"].groupby(quartiles):
        label = []
        q_interval = b.quantile(q=0.75) - b.quantile(q=0.25)
        high = b.quantile(q=0.75) + k * q_interval
        low = b.quantile(q=0.25) - k * q_interval
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

    data = pd.concat([data, s], axis=1)
    data.columns = column_name + ["outlier"]

    # 绘制图像
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 剔除离群点
    normal = data[data["outlier"] == 0]
    outlier = data[data["outlier"] == 1]

    # 正常点与离群点
    plt.scatter(normal["wind_speed"], normal["active_power"], c='g', s=3, alpha=.5, label='normal points')
    plt.scatter(outlier["wind_speed"], outlier["active_power"], c='r', s=3, alpha=.5, label='outliers')
    plt.xlabel('Wind Speed/(m/s)')
    plt.ylabel('Active Power/(kW)')
    plt.legend()
    # plt.savefig('./data/paper/outlier2.png')
    plt.show()

    normal = normal.drop("outlier", axis=1)
    print(normal.describe())
    return normal


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


def WT_modeling(X_tt, Y_tt, X_test, Y_test):
    from sklearn.model_selection import train_test_split

    X_tt = pd.DataFrame(X_tt).values
    Y_tt = pd.DataFrame(Y_tt).values
    # 2017年数据分出一部分作为验证集
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_tt, Y_tt, test_size=0.2, shuffle=None)

    X_test = pd.DataFrame(X_test).values
    Y_test = pd.DataFrame(Y_test).values



    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    models = []
    # 线性回归
    # models.append(("LinearRegression", LinearRegression()))
    models.append(("Ridge", Ridge(alpha=0.6)))
    # models.append(("LogisticRegression", LogisticRegression()))
    models.append(("Lasso", Lasso(alpha=0.002)))
    # 决策树回归
    # models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    # 支持向量回归(误差很大)
    # models.append(("SVR", SVR(C=100000)))
    models.append(("RandomForestRegressor", RandomForestRegressor()))
    # AdaBoostRegressor  base_estimator=DecisionTreeRegressor默认
    # models.append(("AdaBoostRegressor", AdaBoostRegressor()))
    # GBDT 回归
    models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    models.append(("XGBoost", XGBRegressor(max_depth=5, n_estimators=5000)))

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


    cov = np.cov(X_ref.T)
    inv = np.linalg.inv(cov)

    MD = []
    # MD_a = []
    for i in range(len(X_train)):
        md = np.dot(np.dot(delta[i], inv), delta[i].T)
        MD.append(np.sqrt(md))
        # 直接使用函数来计算
        # MD_a.append(mahalanobis(X_ref[i], u, inv))


    # 绘制概率密度直方图
    # sns.distplot(MD, bins=65)
    # plt.show()


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

    pd.Series(MD_app[0:2000]).plot()
    plt.show()


def wt_params(X_train, Y_train):
    """
    xgboost调参函数
    :param X_train:
    :param Y_train:
    scoring评分指标: 'r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error'
                    'neg_mean_squared_log_error'
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    cv_params = {'n_estimators': [500, 550, 450, 480]}
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight':[1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 4,
                    'min_child_weight': 5, 'subsample': 1, 'colsample_bytree': 1,
                    'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error',
                                 cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evaluate_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evaluate_result))
    print('参数的最佳取值: {0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分: {0}'.format(optimized_GBM.best_score_))


def main():
    # 数据预处理
    # filename = './data/year/raw_data2017.csv'
    # raw_data = data_preprocessing(filename)
    # normal_data = DBSCAN_cluster(raw_data, 0.2, 250)
    # data_pre = Quartiles(normal_data, 3, 20)
    # cluster_data = K_Means(raw_data)
    # # draw_clusters(cluster_data)
    # # data_pre.to_csv('./data/year/data_pre2018.csv', index=None)
    # # 增加两个特征
    # data_pre.to_csv('./data/year/data_pre2017_2.csv', index=None)

    # 建模
    # 标准化
    # 使用带前两个时刻温度值的数据集
    data_2017 = './data/year/data_pre2017_2.csv'
    data_2018 = './data/year/data_pre2018.csv'
    X_tt, Y_tt = wt_preprocessing(data_2017)
    X_test, Y_test = wt_preprocessing(data_2018)
    wt_params(X_tt, Y_tt)
    # WT_modeling(X_tt, Y_tt, X_test, Y_test)
    # WT_MD(X_tt, Y_tt)



if __name__ == '__main__':
    main()



'''
6.02
思考:
1.DBSCAN的参数如何选择?尝试k-dist方法,选择的依据
2.目前采用的是基于欧氏距离,效果还可以,尝试马氏距离

6.03
1.特征选择问题,见博客
2.调参:测试集误差较大
xgboost: max_depth=5, n_estimators=5000, learning_rate=0.2
0
XGBoost mean_squared_error 0.726443665234
XGBoost mean_absolute_error 0.627086875969
XGBoost median_absolute_error 0.456092597412

max_depth=7, n_estimators=5000, learning_rate=0.2
过拟合?
0
XGBoost mean_squared_error 0.0050620309286
XGBoost mean_absolute_error 0.0462466675304
XGBoost median_absolute_error 0.026651763916
1
XGBoost mean_squared_error 37.9944184355
XGBoost mean_absolute_error 4.89490600804
XGBoost median_absolute_error 4.0803150177

6.10
测试集误差较大(18年数据)
1.调参
2.增加输入参数维度:其他参数或者前两个时刻的数值
3.限定范围,数据量小一点,几个月的数据
使用18年前三个月数据建立正常模型,预测一个月

'''