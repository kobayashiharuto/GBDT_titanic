import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from pandas.core.frame import DataFrame


class DataProcesser:
    def __init__(self, train, test):
        self.train = train
        self.test = test


# 加工してラベルとデータに分ける
def train_data_treat(train: DataFrame):
    datas = train[['Pclass', 'Sex', 'Age', 'Fare']]
    labels = train['Survived']
    return datas, labels


# 欠損データの確認
def print_lack_table(data_frame: DataFrame):
    null_val_column = data_frame.isnull().sum()
    percent_column = 100 * data_frame.isnull().sum()/len(data_frame)
    lack_table = pd.concat([null_val_column, percent_column], axis=1)
    lack_table_ren_columns = lack_table.rename(
        columns={0: 'Lack', 1: '%'})
    print(lack_table_ren_columns)


if __name__ == '__main__':
    # データ読み込み
    train = pd.read_csv('data/train.csv')
    # datas = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    datas = train

    # 欠損値がある行を削除
    datas = datas.dropna(subset=['Embarked'])

    # 乗船港と性別のカテゴリカルデータを int に変換
    datas['Embarked'] = datas['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    datas['Sex'] = datas['Sex'].map({'male': 0, 'female': 1})

    # キャビン番号をアルファベット順に数値化（アルファベットがAに近づくほど船の上の方に部屋が配置されていたようなので、数値データとして扱う）
    for index, alphabet in enumerate(range(ord('A'), ord('G')+1)):
        alphabet = chr(alphabet)
        datas.loc[datas['Cabin'].str.contains(alphabet, na=False), 'Cabin_Class']\
            = index+1

    # キャビン番号の番号部分だけ取り出す
    datas['Cabin_Num'] = datas['Cabin'].dropna().map(
        lambda x: x.split(' ')[0][1:])

    # キャビンを消す
    datas = datas.drop(columns='Cabin')

    # 名前処理
    datas['Name'] = datas['Name'].dropna().map(
        lambda x: x.split(',')[1].split('.')[0].strip())

    status_mapping = {
        'Mr': 1,
        'Miss': 1,
        'Mrs': 1,
        'Master': 2,
        'Dr': NaN,
        'Rev': NaN,
        'Mlle': 1,
        'Major': NaN,
        'Col': NaN,
        'Mme': 1,
        'Don': 3,
        'Lady': 3,
        'Ms': 1,
        'the Countess': 3,
        'Sir': 2,
        'Jonkheer': 2
    }

    married_mapping = {
        'Mr': NaN,
        'Miss': 0,
        'Mrs': 1,
        'Master': NaN,
        'Dr': NaN,
        'Rev': NaN,
        'Mlle': 1,
        'Major': NaN,
        'Col': NaN,
        'Mme': 1,
        'Don': NaN,
        'Lady': NaN,
        'Ms': 0,
        'the Countess': NaN,
        'Sir': NaN,
        'Jonkheer': NaN,
    }

    # キャプテンを削除
    datas.drop(datas.loc[datas['Name'] == 'Capt'].index, inplace=True)

    datas['Status'] = datas['Name'].map(status_mapping)
    datas['Married'] = datas['Name'].map(married_mapping)

    # 運賃を丸める
    datas['Fare'] = datas['Fare'].round()

    datas.to_csv('data_treated/train.csv')

    print(datas.head(10))
