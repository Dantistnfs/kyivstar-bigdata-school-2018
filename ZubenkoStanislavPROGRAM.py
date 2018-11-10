#!/bin/python
"""
Code for Big Data School 2018
"""
import pandas as pd
import numpy as np
import xgboost as XGB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



# Tweak parameters
tree_depth = 4
num_round = 50
PERCENT_FOR_TRAINING = 80
SHOW_GRAPH = 0
BALANCING_MODE = 0 #no balancing because scoring will be based on AUROCC
# 0 - off
# 1 - upsampling
# 2 - downsampling


def get_balanced_samples(df, col):
    df0 = df[df[col] == 0]
    df1 = df[df[col] == 1]
    print('df1.shape[0]', df1.shape[0])
    return pd.concat([df1, df0.sample(df1.shape[0])], axis=0, ignore_index=True)


if __name__ == "__main__":
    test_set = pd.read_csv("BigDataSchool_test_set.csv")
    train_set = pd.read_csv("BigDataSchool_train_set.csv")
    features = pd.read_csv("BigDataSchool_features.csv")

    # Aggregate user data and create additional variables
    # most of the variables - useless because of small input data

    mean_df = features.groupby("ID").mean().add_suffix('_mean')
    max_df = features.groupby("ID").max().add_suffix('_max')
    sum_df = features.groupby("ID").sum().add_suffix('_sum')
    na_count = features.groupby("ID").apply(lambda x: x.isnull().sum()).add_suffix('_na_count')
    head_df = features.groupby("ID").first()
    tail_df = features.groupby("ID").last()
    abs_diff_df = head_df/tail_df
    abs_diff_df = abs_diff_df.replace(np.nan, 0).replace(np.inf, 0).replace(-np.inf, 0)
    abs_diff_df = abs_diff_df.add_suffix('_abs_diff')
    head_df = head_df.add_suffix('_head')
    tail_df = tail_df.add_suffix('_tail')
    main_df = pd.concat([mean_df, max_df, sum_df, na_count, head_df, tail_df, abs_diff_df], axis=1)


    x_train = train_set.merge(main_df, on="ID", how="left")
    x_output = test_set.merge(main_df, on="ID", how="left")

    y_train = x_train['TARGET']

    x_train.drop(columns=['TARGET'], inplace=True)
    x_output.drop(columns=['TARGET'], inplace=True)

    # divide in train and test df
    c = np.random.randint(100, size=len(x_train))

    x_predict = x_train[c > PERCENT_FOR_TRAINING]
    x_train = x_train[c <= PERCENT_FOR_TRAINING]

    y_predict = y_train[c > PERCENT_FOR_TRAINING]
    y_train = y_train[c <= PERCENT_FOR_TRAINING]

    if BALANCING_MODE == 1:
        sm = SMOTE(random_state=2)
        x_train = x_train.fillna(0)
        x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel()) #try to upsample data
        x_train = pd.DataFrame(x_train_res, columns=x_train.columns)
        y_train = pd.Series(y_train_res)
    if BALANCING_MODE == 2:
        x_train = pd.concat([x_train, pd.DataFrame(y_train, columns=["TARGET"])], axis=1, sort=False)
        x_train = get_balanced_samples(x_train, 'TARGET') #try to downsample data
        y_train = x_train['TARGET']
        x_train.drop(columns=['TARGET'], inplace=True)


    x_train = x_train.drop(columns=["ID"])
    x_predict = x_predict.drop(columns=["ID"])

    x_output_reference = x_output["ID"]
    x_output = x_output.drop(columns=["ID"])

    dtrain = XGB.DMatrix(x_train, label=y_train)
    dpredict = XGB.DMatrix(x_predict)
    doutput = XGB.DMatrix(x_output)
    param = {'max_depth': tree_depth,
                'seed': 52432,
                'booster': 'gbtree',
                'silent': 1,
                'min_child_weight': 1,
                'tree_method': 'approx',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_delta_step' : 1,
                #'scale_pos_weight' : 0.08
                #'subsample': '0.9'
                }
    bst = XGB.train(param.items(), dtrain, num_round)
    if SHOW_GRAPH == 1:
        XGB.plot_tree(bst)
        plt.show()
    # save model
    #bst.save_model('test_model.bin')
    y_pred_XGB_test = bst.predict(dpredict)
    y_pred_XGB_train = bst.predict(dtrain)
    y_pred_XGB_output = bst.predict(doutput)
    predict_roc_XGB = roc_auc_score(y_predict, y_pred_XGB_test)
    train_roc_XGB = roc_auc_score(y_train, y_pred_XGB_train)
    print("Predict accuracy", predict_roc_XGB)
    print("Train accuracy", train_roc_XGB)
    # save result
    result_df = pd.concat([pd.DataFrame(x_output_reference), 
                           pd.DataFrame(y_pred_XGB_output, columns=["TARGET"])], axis=1)
    print(result_df)
    result_df.to_csv("ZubenkoStanislav_test.txt", index=False)
