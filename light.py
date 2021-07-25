import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
import lightgbm as lgb
from data_treat import train_data_treat


# データ読み込み
train = pd.read_csv('data_treated/train.csv')
test = pd.read_csv('data_treated/test.csv')

datas, labels = train_data_treat(train)
print(datas.head(5))
print(labels.head(5))

# データ分割
train_datas, val_datas, train_labels, val_labels =\
    train_test_split(datas, labels, test_size=0.2)

# データ型を変換
lgb_train = lgb.Dataset(train_datas, label=train_labels)
lgb_val = lgb.Dataset(val_datas, label=val_labels)

# パラメータ設定
parameter = {
    'objective': 'binary'
}

# 学習
model = lgb.train(
    params=parameter,
    train_set=lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=10000,
    early_stopping_rounds=100,
    verbose_eval=200
)

# 予測
test_data = test[['Pclass', 'Sex', 'Age', 'Fare']]
predict = model.predict(test_data)
predict = np.where(predict < 0.5, 0, 1)
print(predict)

# グラフ化する
graph = lgb.create_tree_digraph(model, tree_index=0, format='png', name='Tree')
graph.render(view=True)

# CSV に書き出す
id = np.array(test['PassengerId']).astype(int)
prediction = pd.DataFrame(predict, id, columns=['Survived'])
prediction.to_csv('result/lgb_result.csv', index_label=['PassengerId'])
