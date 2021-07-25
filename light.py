import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
import lightgbm as lgb
from data_treat import data_split


# データ読み込み
train = pd.read_csv('data_treated/train.csv')
test = pd.read_csv('data_treated/test.csv')

datas, labels = data_split(train)

print(datas.head(5))
print(labels.head(5))

# データ分割
train_datas, val_datas, train_labels, val_labels =\
    train_test_split(datas, labels, test_size=0.1)

# データ型を変換
lgb_train = lgb.Dataset(train_datas, label=train_labels)
lgb_val = lgb.Dataset(val_datas, label=val_labels)

# パラメータ設定
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 64,
    'min_data_in_leaf': 20,
    'max_depth': 7,
    'verbose': 0,
}

# 学習
model = lgb.train(
    params=params,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=1000,
    early_stopping_rounds=10,
    verbose_eval=10,
    categorical_feature=['Sex', 'Ticket', 'Embarked',
                         'Status', 'Married', 'Doctor', 'Rev', 'Army']
)

# 予測
test_data = test.drop(columns='PassengerId')
predict = model.predict(test_data)
predict = np.where(predict < 0.5, 0, 1)
print(predict)

# グラフ化する
graph = lgb.create_tree_digraph(model, tree_index=0, format='png', name='Tree')
graph.render(directory='out', view=True)

# CSV に書き出す
id = np.array(test['PassengerId']).astype(int)
prediction = pd.DataFrame(predict, id, columns=['Survived'])
prediction.to_csv('result/lgb_result2.csv', index_label=['PassengerId'])
