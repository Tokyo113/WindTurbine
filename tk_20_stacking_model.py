#coding:utf-8
'''
@Time: 2019/6/25 下午2:04
@author: Tokyo
@file: tk_20_stacking_model.py
@desc:  stacking方法建立模型
'''


def stacking_model(features, label):
    import numpy as np
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from mlxtend.regressor import StackingCVRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=0.15, shuffle=False)
    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()


    srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)

    srgr.fit(X_train, Y_train)
    # 五折交叉验证
    # scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    # scores = cross_validate(srgr, X_train, Y_train, scoring=scoring, cv=5)
    # print('MAE', 'Stacking', scores['test_neg_mean_absolute_error'].mean())
    # print('R2', 'Stacking', scores['test_r2'].mean())
    #
    # print('RMSE', 'Stacking', np.sqrt(scores['test_neg_mean_squared_error'] * (-1)).mean())
    # 训练集
    Y_pred1 = srgr.predict(X_train)
    print("RMSE", np.sqrt(mean_squared_error(Y_train, Y_pred1)))
    print("MAE", mean_absolute_error(Y_train, Y_pred1))
    print("r2_score", r2_score(Y_train, Y_pred1))
    # 测试集
    Y_pred = srgr.predict(X_test)
    print("RMSE", np.sqrt(mean_squared_error(Y_test, Y_pred)))
    print("MAE", mean_absolute_error(Y_test, Y_pred))
    print("r2_score", r2_score(Y_test, Y_pred))
    return Y_test, Y_pred


def stacking_model2(X_train, Y_train, X_test, Y_test):
    import numpy as np
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from mlxtend.regressor import StackingCVRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()


    srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)

    srgr.fit(X_train, Y_train)
    # 五折交叉验证
    # scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    # scores = cross_validate(srgr, X_train, Y_train, scoring=scoring, cv=5)
    # print('MAE', 'Stacking', scores['test_neg_mean_absolute_error'].mean())
    # print('R2', 'Stacking', scores['test_r2'].mean())
    #
    # print('RMSE', 'Stacking', np.sqrt(scores['test_neg_mean_squared_error'] * (-1)).mean())
    # 训练集
    # Y_pred1 = srgr.predict(X_train)
    # print("RMSE", np.sqrt(mean_squared_error(Y_train, Y_pred1)))
    # print("MAE", mean_absolute_error(Y_train, Y_pred1))
    # print("r2_score", r2_score(Y_train, Y_pred1))
    # 测试集
    Y_pred = srgr.predict(X_test)
    print("RMSE", np.sqrt(mean_squared_error(Y_test, Y_pred)))
    print("MAE", mean_absolute_error(Y_test, Y_pred))
    print("r2_score", r2_score(Y_test, Y_pred))
    return Y_test, Y_pred


def main():
    import pandas as pd
    import numpy as np
    from tk_18_data_preprocessing import wt_preprocessing
    import matplotlib.pyplot as plt
    from tk_tools import WT_modeling
    import matplotlib.dates as mdate
    # #57号风机
    df1 = pd.read_csv('./data/final data/#57/data2017_half_year_train.csv')
    # #173号风机
    df2 = pd.read_csv('./data/final data/#173/data2018_half_year_train.csv')
    # #159--93号风机
    df3 = pd.read_csv('./data/final data/#159/data2017_half_year_train.csv')
    # df = df.drop(['temp_1', 'temp_2'], axis=1)

    # features, label, names = wt_preprocessing(df, False)
    # stacking_model(features, label)
    # WT_modeling(features, label)

    # 作拟合曲线图
    f = plt.figure(figsize=(8, 8))
    ax1 = f.add_subplot(3, 1, 1)
    features1, label1, names1 = wt_preprocessing(df1, False)
    y_test1, y_pred1 = stacking_model(features1, label1)

    y_pred1 = pd.Series(y_pred1)
    y_pred1.plot(c='g', label='Predict')
    y_test1 = y_test1.reset_index(drop=True)
    # # flatten 降维
    y_test1.plot(c='y', label='Test')
    # 设置时间标签显示格式
    x_ticks = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 4500])
    x_labels = ax1.set_xticklabels(["12/01", "12/7", "12/13", "12/19", "12/25", "12/31"], rotation=30, fontsize="small")
    plt.ylim((20, 85))
    plt.legend(loc='upper left', ncol=2)
    plt.ylabel('Oil Temperature/(Deg.C)')

    ax2 = f.add_subplot(3, 1, 2)
    features2, label2, names2 = wt_preprocessing(df2, False)
    y_test2, y_pred2 = stacking_model(features2, label2)
    y_pred2 = pd.Series(y_pred2)
    y_pred2.plot(c='g', label='Predict')
    y_test2 = y_test2.reset_index(drop=True)
    # # flatten 降维
    y_test2.plot(c='y', label='Test')
    # 设置时间标签显示格式
    x_ticks = ax2.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    x_labels = ax2.set_xticklabels(["12/01", "12/7", "12/13", "12/19", "12/25", "12/31"], rotation=30, fontsize="small")
    plt.legend(loc='upper left', ncol=2)
    plt.ylim((20, 85))
    plt.ylabel('Oil Temperature/(Deg.C)')



    ax3 = f.add_subplot(3, 1, 3)
    features3, label3, names3 = wt_preprocessing(df3, False)
    y_test3, y_pred3 = stacking_model(features3, label3)
    y_pred3 = pd.Series(y_pred3)
    y_pred3.plot(c='g', label='Predict')
    y_test3 = y_test3.reset_index(drop=True)
    # # flatten 降维
    y_test3.plot(c='y', label='Test')
    # 设置时间标签显示格式
    x_ticks = ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    x_labels = ax3.set_xticklabels(["12/01", "12/7", "12/13", "12/19", "12/25", "12/31"], rotation=30, fontsize="small")
    plt.ylim((25, 80))



    plt.xlabel('Date')
    plt.ylabel('Oil Temperature/(Deg.C)')

    plt.legend(loc='upper left', ncol=2)
    # 一定要先保存再show,否则是白板
    plt.savefig('./data/figure/fitting.png', dpi=300)
    plt.show()





if __name__ == '__main__':
    main()
