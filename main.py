#!/bin/python
"""
Code for Big Data School 2018
"""
import pandas as pd
import numpy as np
import xgboost as XGB
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt




# XGBoost parameters
tree_depth = 1
num_round = 50



test_set = pd.read_csv("BigDataSchool_test_set.csv")
train_set = pd.read_csv("BigDataSchool_train_set.csv")
features = pd.read_csv("BigDataSchool_features.csv")



# Aggregate user_data

main_df = features.groupby("ID").sum()
print(main_df)



print(features)



def get_balanced_samples(df, col):
    df0 = df[df[col] == 0]
    df1 = df[df[col] == 1]
    print('df1.shape[0]', df1.shape[0])
    return pd.concat([df1, df0.sample(df1.shape[0])], axis=0, ignore_index=True)

x_train = train_set.merge(main_df, on="ID", how="left")
x_predict = test_set.merge(main_df, on="ID", how="left")
x_train = get_balanced_samples(x_train, 'TARGET') # we need to rebalance because positive data - only 8%
y_train = x_train['TARGET']

x_train.drop(columns=['TARGET'], inplace=True)
x_predict.drop(columns=['TARGET'], inplace=True)

dtrain = XGB.DMatrix(x_train, label=y_train)
dpredict = XGB.DMatrix(x_predict)
param = {'max_depth': tree_depth,
            'seed': 52432,
            'booster': 'gbtree',
            'silent': 1,
            'min_child_weight': 1,
            'tree_method': 'approx',
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
            }
bst = XGB.Booster()
bst = XGB.train(param.items(), dtrain, num_round)
XGB.plot_tree(bst)
plt.show()
# save model
#bst.save_model('models/{0}_{1}.bin'.format(testing_period, run['title']))
y_pred_XGB_test = bst.predict(dpredict)
y_pred_XGB_train = bst.predict(dtrain)
# pred_series.append(pd.Series(y_pred_XGB_test, name=run['title']))
train_roc_XGB = roc_auc_score(y_train, y_pred_XGB_train)
print("Train accuracy", train_roc_XGB)
"""
ft_weights = pd.DataFrame.from_dict([bst.get_fscore()]).T
ft_weights.columns = ['weights']
ft_weights = ft_weights.sort_values(by=['weights'],ascending=False)
ft_weights.to_csv('models/{0}_variable_importance_{1}.csv'.format(testing_period, run['title']))
pd.concat((x_predict_reference_test.reset_index(), pd.Series(y_pred_XGB_train, name=run['title']), y_train.reset_index()), axis=1).drop(['index'], axis=1).to_csv("models/{0}_prediction_on_train_{1}.csv".format(10, run['title']))
pd.concat((x_predict_reference.reset_index(),      pd.Series(y_pred_XGB_test,  name=run['title']), y_test.reset_index()),  axis=1).drop(['index'], axis=1).to_csv("models/{0}_prediction_on_test_{1}.csv".format(10, run['title']))
"""


