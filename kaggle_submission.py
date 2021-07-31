import xgboost as xg
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

#import train.csv
train = pd.read_csv('train.csv')
train_x = train.iloc[:, 1:94]
train_y = train.iloc[:, 94:95]
skb = SelectKBest(mutual_info_classif, k=85)
skb.fit(train_x, train_y.values.ravel())
train_x = skb.transform(train_x)
#import test.csv
test = pd.read_csv('test.csv').iloc[:,1:]

test_x = skb.transform(test)
xgb = xg.XGBClassifier(learning_rate =0.1, n_estimators=450,max_depth=7,min_child_weight=4,
                   objective='multi:softprob', eval_metric='mlogloss')
xgb.fit(train_x,train_y.values.ravel())
pred_y = xgb.predict_proba(test_x)
columns_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
data = pd.DataFrame(pred_y,columns = columns_list)
id = np.arange(1,len(test)+1,1)
data.insert(0, 'id', id)
data.to_csv("G14.csv",index = False)