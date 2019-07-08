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
    X_train, X_validation, Y_train, Y_validation = train_test_split(features, label, test_size=0.15, shuffle=None)
    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()


    srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)

    srgr.fit(X_train, Y_train)
    # 五折交叉验证
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_validate(srgr, X_train, Y_train, scoring=scoring, cv=5)
    print('MAE', 'Stacking', scores['test_neg_mean_absolute_error'].mean())
    print('R2', 'Stacking', scores['test_r2'].mean())

    print('RMSE', 'Stacking', np.sqrt(scores['test_neg_mean_squared_error'] * (-1)).mean())
    # 训练集
    Y_pred1 = srgr.predict(X_train)
    print("RMSE", np.sqrt(mean_squared_error(Y_train, Y_pred1)))
    print("MAE", mean_absolute_error(Y_train, Y_pred1))
    print("r2_score", r2_score(Y_train, Y_pred1))
    # 测试集
    Y_pred = srgr.predict(X_validation)
    print("RMSE", np.sqrt(mean_squared_error(Y_validation, Y_pred)))
    print("MAE", mean_absolute_error(Y_validation, Y_pred))
    print("r2_score", r2_score(Y_validation, Y_pred))


def main():
    import pandas as pd
    import numpy as np
    from tk_18_data_preprocessing import wt_preprocessing
    from tk_tools import WT_modeling
    # #57号风机
    df = pd.read_csv('./data/final data/#57/data2017_half_year_train.csv')
    # #173号风机
    # df = pd.read_csv('./data/final data/#173/data2018_half_year_train.csv')
    # df = df.drop(['temp_1', 'temp_2'], axis=1)

    features, label, names = wt_preprocessing(df, False)
    stacking_model(features, label)
    # WT_modeling(features, label)



if __name__ == '__main__':
    main()
