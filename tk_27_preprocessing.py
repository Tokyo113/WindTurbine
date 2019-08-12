#coding:utf-8
'''
@Time: 2019/8/12 下午5:14
@author: Tokyo
@file: tk_27_preprocessing.py
@desc: https://blog.csdn.net/qq_40304090/article/details/90597892
之前的标准化,归一化是不对的,应该先对训练集标准化,然后将规则应用于测试集
'''


def wt_preprocessing(train_data, test_data, method):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    train_data = train_data.drop("date", axis=1)
    train_data = train_data.drop("state", axis=1)
    test_data = test_data.drop("date", axis=1)
    test_data = test_data.drop("state", axis=1)

    # 得到特征和标签
    label = train_data["Gearbox_oil_temp"]
    label_te = test_data["Gearbox_oil_temp"]

    features = train_data.drop("Gearbox_oil_temp", axis=1)
    test_data = test_data.drop("Gearbox_oil_temp", axis=1)

    if method:
        std_minmax = MinMaxScaler().fit(features)
        features = std_minmax.transform(features)
        data_te = std_minmax.transform(test_data)
    else:
        std_sta = StandardScaler().fit(features)
        features = std_sta.transform(features)
        data_te = std_sta.transform(test_data)
        features = StandardScaler().fit_transform(features)
    return features, label, data_te, label_te


def main():
    pass



if __name__ == '__main__':
    main()