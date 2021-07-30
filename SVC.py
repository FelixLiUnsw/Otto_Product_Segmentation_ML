# Template
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC,LinearSVC
np.random.seed(12)
from sklearn.metrics import accuracy_score

# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:, 1:94]
data_y = data.iloc[:, 94:95]

# test data import
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40)

# ---------
columns_list = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]
yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns=columns_list)
for i in range(len(valid_y)):
    index = columns_list.index(valid_y.iloc[i].item())
    yij.iloc[i, index] = 1

# Multi-Class Log-Loss
def logloss(pred_y, y_ij):
    max_p = 1 - 1e-15
    min_p = 1e-15
    num_of_Products = len(pred_y)

    pred_y[(pred_y >= max_p)] = max_p
    pred_y[(pred_y <= min_p)] = min_p

    #print(pred_y)
    log_prob = np.log(pred_y)
    y_multi_prob = np.multiply(y_ij, log_prob)
    #print("y_multi_prob: ", y_multi_prob)
    y_multi_prob = np.array(y_multi_prob)
    sum_value = np.sum(y_multi_prob)
    #print("Sum_value is: ", sum_value)
    log_loss_value = -np.divide(sum_value, num_of_Products)
    return log_loss_value

# TODO -----------
start=time.perf_counter()

clf=SVC(C = 1.0, kernel='rbf',probability=True)
clf.fit(train_x,train_y.values.ravel())
pred_y=clf.predict_proba(valid_x)
pred_y2=clf.predict(valid_x)
score=accuracy_score(valid_y,pred_y2)
print('c=',1,"accuracy:",score)
Multi_class_log_loss = logloss(pred_y, yij)
print("The Multi Class Log Loss: ", Multi_class_log_loss)
end = time.perf_counter()
print("running time: %s Senconds" % (end - start))
print()



