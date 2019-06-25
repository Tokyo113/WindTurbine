#coding:utf-8
'''
@Time: 2019/6/25 下午2:04
@author: Tokyo
@file: tk_20_stacking_model.py
@desc:  stacking方法建立模型
'''


def stacking_model(features, label):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from mlxtend.regressor import StackingCVRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    X_train, X_validation, Y_train, Y_validation = train_test_split(features, label, test_size=0.2, shuffle=None)
    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    lr = Ridge()

    srgr = StackingCVRegressor(regressors=[xgb, rfr], meta_regressor=lr)

    srgr.fit(X_train, Y_train)
    Y_pred = srgr.predict(X_validation)
    print("mean_squared_error", mean_squared_error(Y_validation, Y_pred))
    print("mean_absolute_error", mean_absolute_error(Y_validation, Y_pred))
    print("r2_score", r2_score(Y_validation, Y_pred))


def main():
    import pandas as pd
    import numpy as np
    from tk_18_data_preprocessing import wt_preprocessing
    df = pd.read_csv('./data/data2018_half_year_train.csv')
    # df = df.drop(['temp_1', 'temp_2'], axis=1)

    features, label, names = wt_preprocessing(df, False)
    stacking_model(features, label)




if __name__ == '__main__':
    main()