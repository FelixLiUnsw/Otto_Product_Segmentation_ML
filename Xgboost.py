import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xg
import datetime

start = datetime.datetime.now()
def y_ij():
    columns_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns = columns_list)
    for i in range(len(valid_y)):
        index = columns_list.index(valid_y.iloc[i].item())
        yij.iloc[i,index] = 1
    return yij

def logloss(pred_y, y_ij):
    max_p = 1 - 1e-15
    min_p = 1e-15
    num_of_Products = len(pred_y)

    pred_y[(pred_y >= max_p)] = max_p
    pred_y[(pred_y <= min_p)] = min_p

    log_prob = np.log(pred_y)
    y_multi_prob = np.multiply(y_ij, log_prob)
    y_multi_prob = np.array(y_multi_prob)
    sum_value = np.sum(y_multi_prob)
    log_loss_value = -np.divide(sum_value, num_of_Products)
    return log_loss_value


np.random.seed(12)
# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:, 1:94]
data_y = data.iloc[:, 94:95]

train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40, )

yij = y_ij()

# optimize n_estimators
scores = []
n_estimators = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]
for nes in n_estimators:
    xgb = xg.XGBClassifier(learning_rate=0.1, n_estimators=nes, max_depth=6, min_child_weight=1,
                           objective='multi:softprob', eval_metric='mlogloss')
    xgb.fit(train_x, train_y.values.ravel())
    pred_y = xgb.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
plt.plot(n_estimators, scores, 'o-')
plt.ylabel(logloss)
plt.xlabel("n_estimator")
plt.savefig('./n_estimator.png')
print("best n_estimator {}".format(n_estimators[np.argmin(scores)]))
plt.show()

# optimize max_depth
scores_md = []
max_depths = [1,3,5,6,7,8,10]
for md in max_depths:
    xgb = xg.XGBClassifier(learning_rate =0.1, n_estimators=450,
                           max_depth=md, min_child_weight=1, objective='multi:softprob', eval_metric='mlogloss')
    xgb.fit(train_x, train_y.values.ravel())
    pred_y = xgb.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores_md.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
plt.plot(max_depths,scores_md,'o-')
plt.ylabel(logloss)
plt.xlabel("max_depth")
print("best max_depth {}".format(max_depths[np.argmin(scores_md)]))
plt.show()

# optimize eta
scores_eta = []
etas = [0.001,0.01,0.1,0.2,0.3,0.5,1]
for eta in etas:
    xgb = xg.XGBClassifier(learning_rate =eta, n_estimators=450,
                           max_depth=7,
                           min_child_weight=1, objective='multi:softprob', eval_metric='mlogloss')
    xgb.fit(train_x, train_y.values.ravel())
    pred_y = xgb.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores_eta.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
plt.plot(etas,scores_eta,"o-")
plt.ylabel(logloss)
plt.xlabel("eta")
print("best eta {}".format(etas[np.argmin(scores_eta)]))
plt.show()

# optimize min_child_weight
scores_mcw = []
min_child_weights = [1,2,3,4,5,6,7,8]
for mcw in min_child_weights:
    xgb = xg.XGBClassifier(learning_rate=0.1, n_estimators=450,
                           max_depth=7,
                           min_child_weight=mcw, objective='multi:softprob', eval_metric='mlogloss')
    xgb.fit(train_x, train_y.values.ravel())
    pred_y = xgb.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores_mcw.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
plt.plot(min_child_weights,scores_mcw)
plt.ylabel(logloss)
plt.xlabel("min_child_weight")
print("best min_child_weight {}".format(min_child_weights[np.argmin(scores_mcw)]))
plt.show()

# the best parameters
xgb = xg.XGBClassifier(learning_rate =0.1, n_estimators=450,max_depth=7,min_child_weight=4,
                       objective='multi:softprob', eval_metric='mlogloss')
xgb.fit(train_x, train_y.values.ravel())
pred_y = xgb.predict_proba(valid_x)
print("The Log Loss is: ", logloss(pred_y, yij))
#print("Accuracy ", xgb.score(valid_x, valid_y))
end = datetime.datetime.now()
print('Duration:', end-start)
