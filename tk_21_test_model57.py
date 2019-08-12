#coding:utf-8
'''
@Time: 2019/7/10 下午4:02
@author: Tokyo
@file: tk_21_test_model57.py
@desc:实例分析---#57号风机,一个月数据为测试集
'''

def stacking_MD(X_train, Y_train, X_test, Y_test):
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import pdist, mahalanobis
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from mlxtend.regressor import StackingCVRegressor
    from sklearn.externals import joblib
    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()
    # stacking model
    # srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)
    # srgr.fit(X_train, Y_train)

    # 直接读取保存好的模型 stacking.pkl---type C
    srgr = joblib.load('stacking.pkl')


    # 计算马氏距离
    # 1.训练集,求得马氏距离检测异常的阈值
    Y_pred = srgr.predict(X_train)
    # Y_train = Y_train.flatten()
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
    for i in range(len(X_train)):
        MD.append(mahalanobis(X_ref[i], u, inv))

    # 2.验证集(用于异常检测)
    Y_pred2 = srgr.predict(X_test)
    # Y_test = Y_test.flatten()

    # 计算马氏距离
    err_test = Y_test - Y_pred2

    X_app = np.array([err_test, Y_test])
    X_app = X_app.T
    # 均值u和协方差矩阵inv均使用训练集得出的值
    MD_app = []
    for i in range(len(X_test)):
        MD_app.append(mahalanobis(X_app[i], u, inv))
    md = pd.Series(MD_app)
    return md



def main():
    import pandas as pd
    import numpy as np
    from tk_18_data_preprocessing import wt_preprocessing
    import matplotlib.pyplot as plt
    from tk_tools import wt_Cusum_change_point_detection
    df_train = pd.read_csv('./data/final data/#57/data2017_half_year_train.csv')
    X_train, Y_train, names = wt_preprocessing(df_train, False)
    df_test = pd.read_csv('./data/final data/#57/data2018_Jan_test2.csv')
    # df_test = df_test.drop_duplicates(subset=['date'])

    X_test, Y_test, names2 = wt_preprocessing(df_test, False)
    md_57 = stacking_MD(X_train, Y_train, X_test, Y_test)
    # wt_Cusum_change_point_detection(md_57, 1000, 0.99)
    f = plt.figure(figsize=(8, 8))

    # 计算Cusum序列
    arr = np.array(md_57)
    s = np.zeros(len(arr) + 1)
    for i in range(1, len(arr)):
        s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
    s = pd.Series(s)
    # 马氏距离序列图
    ax1 = f.add_subplot(3, 1, 1)
    md_57.plot()
    x_ticks = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5900])
    x_labels = ax1.set_xticklabels(["Jan/01", "Jan/05", "Jan/11", "Jan/18", "Jan/21", "Jan/27",  "Feb/01"],
                                   rotation=30, fontsize="small")
    plt.ylabel('Mahalanobis Distance')

    ax2 = f.add_subplot(3, 1, 2)
    s.plot()
    x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5900])
    x_labels = ax2.set_xticklabels(["Jan/01", "Jan/05", "Jan/11", "Jan/18", "Jan/21", "Jan/27", "Feb/01"],
                                   rotation=30, fontsize="small")
    plt.ylabel('CUSUM chart')
    # 加注释
    plt.annotate('Change Point 1', xy=(488, s[488]), xycoords='data',
                 xytext=(-30, -70), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate('Change Point 2', xy=(1265, s[1265]), xycoords='data',
                 xytext=(+10, -40), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Change Point 3', xy=(2567, s[2567]), xycoords='data',
                 xytext=(+25, -5), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Change Point 4', xy=(4252, s[4252]), xycoords='data',
                 xytext=(+25, +10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    ax3 = f.add_subplot(3, 1, 3)
    df_oil = df_test['Gearbox_oil_temp']
    df_oil.plot(label='Measured Temperature')
    x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5900])
    x_labels = ax3.set_xticklabels(["Jan/01", "Jan/05", "Jan/11", "Jan/18", "Jan/21", "Jan/27", "Feb/01"],
                                   rotation=30, fontsize="small")
    plt.ylabel('Oil Temperature/(Deg.C)')
    plt.xlabel('Date')
    plt.hlines(y=80, xmin=0, xmax=5900, colors='r', label='Upper Limit')
    plt.legend(ncol=2)
    plt.savefig('./data/figure/#57result.png', dpi=300)
    plt.show()





if __name__ == '__main__':
    main()