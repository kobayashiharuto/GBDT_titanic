import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


# データ読み込み
train = pd.read_csv('data/titanic_train_treated.csv')
test = pd.read_csv('data/titanic_test_treated.csv')

# 説明変数と目的変数に分ける
data = train[['Pclass', 'Sex', 'Age', 'Fare']]
target = train['Survived']


# データ分割
train_datas, test_datas, train_labels, test_labels = train_test_split(
    data, target, test_size=0.3)

# XGBoost用のデータ型に変換する
xg_train = xgb.DMatrix(train_datas, label=train_labels)
xg_test = xgb.DMatrix(test_datas, label=test_labels)

# パラメーター設定
params = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'random_state': 1234,
    'eval_metric': 'rmse',
}

# 訓練用データと評価用データを適用
watchlist = [(xg_train, 'train'), (xg_test, 'eval')]

# モデル訓練
model = xgb.train(
    params,
    xg_train,
    num_boost_round=50,
    early_stopping_rounds=20,
    evals=watchlist,
)

# 変数重要度を出力
mapper = {f'f{i}': v for i, v in enumerate(['Pclass', 'Sex', 'Age', 'Fare'])}
mapped = {mapper[k]: v for k, v in model.get_fscore().items()}
xgb.plot_importance(mapped)

# 決定木を可視化する
graph = xgb.to_graphviz(model)
graph.format = 'png'
graph.render('out/XGtree1')

# 予測
predict_tagert = xgb.DMatrix(test[['Pclass', 'Sex', 'Age', 'Fare']].values)
predict = model.predict(predict_tagert, ntree_limit=model.best_ntree_limit)

# データを整形して書き出し
predict = np.round(predict).astype(int)
id = np.array(test['PassengerId']).astype(int)

result = pd.DataFrame(predict, id, columns=['Survived'])
result.to_csv('result/xgb_result.csv', index_label=['PassengerId'])
