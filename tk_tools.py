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
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.15, shuffle=False)


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
    # md[0:200].rolling(6).mean().plot()

    plt.show()
    return md

def wt_s_diff(arr):
    """
    计算最大距离
    :param arr:输入数据
    :return: 返回最大距离
    """
    s = np.zeros(len(arr) + 1)
    for i in range(1, len(arr)):
        s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
    return s.max() - s.min()


def wt_Cusum_change_point_detection(inputdata, n, confi):
    """
    Cusum控制图变点检测函数
    :param inputdata: 输入数据
    :param n: 采样次数
    :param confi: 置信度
    :return: 变点的索引
    """
    arr = np.array(inputdata)
    s = np.zeros(len(arr) + 1)
    for i in range(1, len(arr)):
        s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
    # 最大距离
    arr_diff = s.max() - s.min()
    # 最大S值(绝对值)的索引
    arr_index = np.argmax(abs(s))
    count = 0
    for i in range(0, n):
        bs_sample = np.random.choice(inputdata, size=len(inputdata), replace=False)
        s_diff = wt_s_diff(bs_sample)
        if s_diff < arr_diff:
            count += 1
    confidence_level = count / n
    if confidence_level > confi:
        print(arr_index)
        wt_Cusum_change_point_detection(arr[0:arr_index], n, confi)
        # wt_Cusum_change_point_detection(arr[arr_index+1:], n, confi)

    return



def Pettitt_change_point_detection(inputdata):
    print(len(inputdata))
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2*np.sum(r[0:x])-x*(n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U**2))/(n**3 + n**2))
    if pvalue <= 0.05:
        change_point_desc = '显著'
    else:
        change_point_desc = '不显著'
    # Pettitt_result = {'突变点位置':K,'突变程度':change_point_desc}
    return K





def Kendall_change_point_detection(inputdata):
    # Mann-Kendall突变点检测
    # 数据序列y
    # 结果序列UF，UB
    # --------------------------------------------

    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
    # 正序列计算---------------------------------
    # 定义累计量序列Sk，初始值=0
    Sk = [0]
    # 定义统计量UFk，初始值 =0
    UFk = [0]
    # 定义Sk序列元素s，初始值 =0
    s = 0
    Exp_value = [0]
    Var_value = [0]
    # i从1开始，因为根据统计量UFk公式，i=0时，Sk(0)、E(0)、Var(0)均为0
    # 此时UFk无意义，因此公式中，令UFk(0)=0
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        Exp_value.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UFk.append((Sk[i]-Exp_value[i])/np.sqrt(Var_value[i]))
    # ------------------------------正序列计算
    # 逆序列计算---------------------------------
    # 定义逆序累计量序列Sk2，长度与inputdata一致，初始值=0
    Sk2 = [0]
    # 定义逆序统计量UBk，长度与inputdata一致，初始值=0
    UBk = [0]
    UBk2 = [0]
    # s归0
    s2 =  0
    Exp_value2 = [0]
    Var_value2 = [0]
    # 按时间序列逆转样本y
    inputdataT = list(reversed(inputdata))
    # i从2开始，因为根据统计量UBk公式，i=1时，Sk2(1)、E(1)、Var(1)均为0
    # 此时UBk无意义，因此公式中，令UBk(1)=0
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2+1
            else:
                s2 = s2+0
        Sk2.append(s2)
        Exp_value2.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value2.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UBk.append((Sk2[i]-Exp_value2[i])/np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    # 由于对逆序序列的累计量Sk2的构建中，依然用的是累加法，即后者大于前者时s加1，
    # 则s的大小表征了一种上升的趋势的大小，而序列逆序以后，应当表现出与原序列相反
    # 的趋势表现，因此，用累加法统计Sk2序列，统计量公式(S(i)-E(i))/sqrt(Var(i))
    # 也不应改变，但统计量UBk应取相反数以表征正确的逆序序列的趋势
    #  UBk(i)=0-(Sk2(i)-E)/sqrt(Var)
    # ------------------------------逆序列计算
    # 此时上一步的到UBk表现的是逆序列在逆序时间上的趋势统计量
    # 与UFk做图寻找突变点时，2条曲线应具有同样的时间轴，因此
    # 再按时间序列逆转结果统计量UBk，得到时间正序的UBkT，
    UBkT = list(reversed(UBk2))
    diff = np.array(UFk) - np.array(UBkT)
    K = list()
    # 找出交叉点
    for k in range(1,n):
        if diff[k-1]*diff[k]<0:
            K.append(k)
    # 做突变检测图时，使用UFk和UBkT
    plt.figure(figsize=(10,5))
    plt.plot(range(1,n+1), UFk, label='UFk') # UFk
    plt.plot(range(1,n+1), UBkT, label='UBk') # UBk
    plt.ylabel('UFk-UBk')
    x_lim = plt.xlim()
    plt.plot(x_lim, [-1.96, -1.96], 'm--', color='r')
    plt.plot(x_lim, [0,  0], 'm--')
    plt.plot(x_lim, [+1.96, +1.96], 'm--', color='r')
    plt.legend(loc=2) # 图例
    plt.show()
    return K


def WT_modeling(features, label):
    """
    回归模型对比,带交叉验证
    :param features:
    :param label:
    """
    from sklearn.model_selection import train_test_split
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values

    # 切分训练集,测试集,验证集
    # X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.15)
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    models = []

    models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    # models.append(("RandomForestRegressor", RandomForestRegressor()))
    models.append(("SVR", SVR()))
    models.append(("AdaBoostRegressor", AdaBoostRegressor()))

    # GBDT 回归
    # models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    # models.append(("XGBoost", XGBRegressor()))
    for regr_name, regr in models:
        regr.fit(X_train, Y_train)
        # 五折交叉验证
        # scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        # scores = cross_validate(regr, X_train, Y_train, scoring=scoring, cv=5)
        # print('MAE', regr_name,  scores['test_neg_mean_absolute_error'].mean())
        # print('R2', regr_name, scores['test_r2'].mean())
        #
        # print('RMSE', regr_name, np.sqrt(scores['test_neg_mean_squared_error']*(-1)).mean())

        xy_lst = [(X_train, Y_train), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = regr.predict(X_part)

            print(i)
            # 0--训练集, 1--测试集
            print(regr_name, "RMSE", np.sqrt(mean_squared_error(Y_part, Y_pred)))
            print(regr_name, "MAE", mean_absolute_error(Y_part, Y_pred))
            print(regr_name, "R2", r2_score(Y_part, Y_pred))


def WT_modeling(features, label):
    """
    回归模型对比,带交叉验证
    :param features:
    :param label:
    """
    from sklearn.model_selection import train_test_split
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values

    # 切分训练集,测试集,验证集
    # X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.15)
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    models = []

    models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    # models.append(("RandomForestRegressor", RandomForestRegressor()))
    models.append(("SVR", SVR()))
    models.append(("AdaBoostRegressor", AdaBoostRegressor()))

    # GBDT 回归
    # models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    # models.append(("XGBoost", XGBRegressor()))
    for regr_name, regr in models:
        regr.fit(X_train, Y_train)
        # 五折交叉验证
        # scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        # scores = cross_validate(regr, X_train, Y_train, scoring=scoring, cv=5)
        # print('MAE', regr_name,  scores['test_neg_mean_absolute_error'].mean())
        # print('R2', regr_name, scores['test_r2'].mean())
        #
        # print('RMSE', regr_name, np.sqrt(scores['test_neg_mean_squared_error']*(-1)).mean())

        xy_lst = [(X_train, Y_train), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = regr.predict(X_part)

            print(i)
            # 0--训练集, 1--测试集
            print(regr_name, "RMSE", np.sqrt(mean_squared_error(Y_part, Y_pred)))
            print(regr_name, "MAE", mean_absolute_error(Y_part, Y_pred))
            print(regr_name, "R2", r2_score(Y_part, Y_pred))

def WT_modeling2(features, label, X_test, Y_test):
    """
    回归模型对比,四个输入参数
    :param features:
    :param label:
    """
    from sklearn.model_selection import train_test_split
    f_v = pd.DataFrame(features).values
    f_names = pd.DataFrame(features).columns.values
    l_v = pd.DataFrame(label).values
    X_train = f_v
    Y_train = l_v

    # 切分训练集,测试集,验证集
    # X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
    # X_train, X_test, Y_train, Y_test = train_test_split(f_v, l_v, test_size=0.15)
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    models = []

    models.append(("DecisionTreeRegressor", DecisionTreeRegressor()))
    models.append(("RandomForestRegressor", RandomForestRegressor()))
    models.append(("SVR", SVR()))
    models.append(("AdaBoostRegressor", AdaBoostRegressor()))
    models.append(("KNN", KNeighborsRegressor()))

    # GBDT 回归
    # models.append(("GradientBoostingRegressor", GradientBoostingRegressor()))
    # XGBoost
    # models.append(("XGBoost", XGBRegressor()))
    for regr_name, regr in models:
        regr.fit(X_train, Y_train)
        # 五折交叉验证
        # scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        # scores = cross_validate(regr, X_train, Y_train, scoring=scoring, cv=5)
        # print('MAE', regr_name,  scores['test_neg_mean_absolute_error'].mean())
        # print('R2', regr_name, scores['test_r2'].mean())
        #
        # print('RMSE', regr_name, np.sqrt(scores['test_neg_mean_squared_error']*(-1)).mean())

        xy_lst = [(X_train, Y_train), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = regr.predict(X_part)

            print(i)
            # 0--训练集, 1--测试集
            print(regr_name, "RMSE", np.sqrt(mean_squared_error(Y_part, Y_pred)))
            print(regr_name, "MAE", mean_absolute_error(Y_part, Y_pred))
            print(regr_name, "R2", r2_score(Y_part, Y_pred))


def main():
    input = [10.7, 13, 11.4, 11.5, 12.5, 14.1, 14.8, 14.1, 12.6, 16, 11.7, 10.6,
             10, 11.4, 7.9, 9.5, 8.0, 11.8, 10.5, 11.2, 9.2, 10.1, 10.4, 10.5]
    input2 = [1, 2, 3.5, 4.6]
    input = np.array(input)
    # 三种变点检测方法
    wt_Cusum_change_point_detection(input, 10000, 0.95)
    print(Pettitt_change_point_detection(input))
    print(Kendall_change_point_detection(input))




if __name__ == '__main__':
    main()