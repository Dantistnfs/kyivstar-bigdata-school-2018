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



# XGBoost parameters
tree_depth = 4
num_round = 50


PERCENT_FOR_TRAINING = 90
BALANCING_MODE = 0
# 0 - off
# 1 - upsampling
# 2 - downsampling


def get_balanced_samples(df, col):
    df0 = df[df[col] == 0]
    df1 = df[df[col] == 1]
    print('df1.shape[0]', df1.shape[0])
    return pd.concat([df1, df0.sample(df1.shape[0])], axis=0, ignore_index=True)


test_set = pd.read_csv("BigDataSchool_test_set.csv")
train_set = pd.read_csv("BigDataSchool_train_set.csv")
features = pd.read_csv("BigDataSchool_features.csv")

# Aggregate user data and create additional variables
# most of the variables - useless because of small input data

mean_df = features.groupby("ID").mean().add_suffix('_mean')
max_df = features.groupby("ID").max().add_suffix('_max')
sum_df = features.groupby("ID").sum().add_suffix('_sum')
na_count = features.groupby("ID").apply(lambda x: x.isnull().sum()).add_suffix('_na_count')
head_df = features.groupby("ID").first().add_suffix('_head')
tail_df = features.groupby("ID").last().add_suffix('_tail')
main_df = pd.concat([mean_df,max_df,sum_df, na_count, head_df, tail_df], axis=1)


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
predict_roc_XGB = roc_auc_score(y_predict, y_pred_XGB_test)
train_roc_XGB = roc_auc_score(y_train, y_pred_XGB_train)
print("Predict accuracy", predict_roc_XGB)
print("Train accuracy", train_roc_XGB)
"""
ft_weights = pd.DataFrame.from_dict([bst.get_fscore()]).T
ft_weights.columns = ['weights']
ft_weights = ft_weights.sort_values(by=['weights'],ascending=False)
ft_weights.to_csv('models/{0}_variable_importance_{1}.csv'.format(testing_period, run['title']))
pd.concat((x_predict_reference_test.reset_index(), pd.Series(y_pred_XGB_train, name=run['title']), y_train.reset_index()), axis=1).drop(['index'], axis=1).to_csv("models/{0}_prediction_on_train_{1}.csv".format(10, run['title']))
pd.concat((x_predict_reference.reset_index(),      pd.Series(y_pred_XGB_test,  name=run['title']), y_test.reset_index()),  axis=1).drop(['index'], axis=1).to_csv("models/{0}_prediction_on_test_{1}.csv".format(10, run['title']))
"""


