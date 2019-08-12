#coding:utf-8
'''
@Time: 2019/8/12 下午8:11
@author: Tokyo
@file: tk_28_saveModel.py
@desc: 保存模型
'''



def save_model(X_train, Y_train):

    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from mlxtend.regressor import StackingCVRegressor
    from sklearn.externals import joblib


    xgb = XGBRegressor()
    rfr = RandomForestRegressor()
    gbdt = GradientBoostingRegressor()

    srgr = StackingCVRegressor(regressors=[xgb, rfr, gbdt], meta_regressor=xgb, cv=5)

    srgr.fit(X_train, Y_train)
    joblib.dump(srgr, 'stacking.pkl')



def main():
    import pandas as pd
    import numpy as np
    from tk_19_feature_selection import feature_selection
    from tk_20_stacking_model import stacking_model, stacking_model2
    from tk_27_preprocessing import wt_preprocessing
    from tk_21_test_model57 import stacking_MD

    df1 = pd.read_csv('./data/C/train/training set.csv')
    df1 = df1.dropna()
    df1 = feature_selection(df1)

    df1_t = pd.read_csv('./data/C/hfj061_28/validation set.csv')
    df1_t = df1_t.dropna()
    df1_t = feature_selection(df1_t)

    features, label, data_te, label_te = wt_preprocessing(df1, df1_t, False)
    save_model(features, label)


if __name__ == '__main__':
    main()