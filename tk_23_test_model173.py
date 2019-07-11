#coding:utf-8
'''
@Time: 2019/7/11 下午2:01
@author: Tokyo
@file: tk_23_test_model173.py
@desc:#173号风机最终测试
'''

def stacking_MD(X_train, Y_train, X_test, Y_test):
    import pandas as pd
    import numpy as np
    from scipy.spatial.distance import pdist, mahalanobis
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from mlxtend.regressor import StackingCVRegressor

    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()
    # stacking model
    srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)
    srgr.fit(X_train, Y_train)
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
    df_train = pd.read_csv('./data/final data/#173/data2018_half_year_train.csv')
    X_train, Y_train, names = wt_preprocessing(df_train, False)
    df_test = pd.read_csv('./data/final data/#173/data2018_April_test.csv')
    df_test = df_test.drop_duplicates(subset=['date'])

    X_test, Y_test, names2 = wt_preprocessing(df_test, False)
    md_159 = stacking_MD(X_train, Y_train, X_test, Y_test)
    # wt_Cusum_change_point_detection(md_159, 1000, 0.99)
    f = plt.figure(figsize=(8, 8))

    # 计算Cusum序列
    arr = np.array(md_159)
    s = np.zeros(len(arr) + 1)
    for i in range(1, len(arr)):
        s[i] = s[i - 1] + (arr[i - 1] - arr.mean())
    s = pd.Series(s)
    # 马氏距离序列图
    ax1 = f.add_subplot(3, 1, 1)
    md_159.plot()
    x_ticks = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5700])
    x_labels = ax1.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21", "Apr/27",  "Apr/30"],
                                   rotation=30, fontsize="small")
    plt.ylabel('Mahalanobis Distance')

    ax2 = f.add_subplot(3, 1, 2)
    s.plot()
    x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5700])
    x_labels = ax2.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21", "Apr/27",  "Apr/30"],
                                   rotation=30, fontsize="small")
    plt.ylabel('CUSUM chart')
    # 加注释
    plt.annotate('Change Point 1', xy=(752, s[752]), xycoords='data',
                 xytext=(+5, +50), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Change Point 2', xy=(3299, s[3299]), xycoords='data',
                 xytext=(+10, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    plt.annotate('Change Point 3', xy=(3716, s[3716]), xycoords='data',
                 xytext=(+25, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    # plt.annotate('Change Point 4', xy=(4252, s[4252]), xycoords='data',
    #              xytext=(+25, +10), textcoords='offset points',
    #              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    ax3 = f.add_subplot(3, 1, 3)
    df_oil = df_test['Gearbox_oil_temp']
    df_oil.plot(label='Measured Temperature')
    x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 5700])
    x_labels = ax3.set_xticklabels(["Apr/01", "Apr/05", "Apr/10", "Apr/15", "Apr/21", "Apr/27",  "Apr/30"],
                                   rotation=30, fontsize="small")
    plt.ylabel('Oil Temperature/(Deg.C)')
    plt.xlabel('Date')
    plt.hlines(y=80, xmin=0, xmax=6200, colors='r', label='Upper Limit')
    plt.legend(loc='lower left', ncol=2)
    plt.savefig('./data/figure/#173result.png', dpi=300)
    plt.show()





if __name__ == '__main__':
    main()