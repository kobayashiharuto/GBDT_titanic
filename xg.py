import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


train = pd.read_csv('data/titanic_train_treated.csv')
test = pd.read_csv('data/titanic_test_treated.csv')
