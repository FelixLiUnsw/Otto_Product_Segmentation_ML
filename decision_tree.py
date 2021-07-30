import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import datetime

start = datetime.datetime.now()

def y_generate():
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

yij = y_generate()

scores_md = []
max_depths = [5, 10, 15, 20, 25, 30]
for md in max_depths:
    clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=5)
    clf.fit(train_x, train_y)
    pred_y = clf.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores_md.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
# plt.figure(1)
# plt.plot(max_depths,scores_md)
# plt.ylabel(logloss)
# plt.xlabel("max_depth")
print("best max_depth {}".format(max_depths[np.argmin(scores_md)]))



scores_msl = []
min_samples_leafs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
for mls in min_samples_leafs:
    clf = DecisionTreeClassifier(max_depth=max_depths[np.argmin(scores_md)], min_samples_leaf=mls)
    clf.fit(train_x, train_y)
    pred_y = clf.predict_proba(valid_x)
    score = logloss(pred_y, yij)
    scores_msl.append(score)
    print("test data log loss eval : {}".format(logloss(pred_y, yij)))
# plt.figure(2)
# plt.plot(min_samples_leafs,scores_msl)
# plt.ylabel(logloss)
# plt.xlabel("min_samples_leaf")
print("best min_samples_leaf {}".format(min_samples_leafs[np.argmin(scores_msl)]))
# plt.show()

clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=9)
clf.fit(train_x,train_y)
pred_y = clf.predict_proba(valid_x)
print("The Log Loss is: ", logloss(pred_y, yij))
print(clf.score(valid_x,valid_y))
end = datetime.datetime.now()
print('Duration:', end-start)