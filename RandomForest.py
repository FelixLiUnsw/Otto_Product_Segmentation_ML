# Random Forest Classification
# Jiaxuan Li z5086369
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def logloss(pred_y, y_ij):
    max_p = 1-1e-15
    min_p = 1e-15
    num_of_Products = len(pred_y)
    
    pred_y[(pred_y >= max_p)] = max_p
    pred_y[(pred_y <= min_p)] = min_p
    
    log_prob = np.log(pred_y)
    y_multi_prob = np.multiply(y_ij,log_prob)
    y_multi_prob = np.array(y_multi_prob)
    sum_value = np.sum(y_multi_prob)
    log_loss_value = -np.divide(sum_value,num_of_Products)
    return log_loss_value

np.random.seed(12)
# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:,1:94]
data_y = data.iloc[:,94:95]

#test data import
train_x, valid_x, train_y, valid_y=train_test_split(data_x,data_y,test_size=0.3,random_state=40)

#Random Forest Classification

trees = np.linspace(25, 300, 12)
print(trees[0])
pred_y_list = []
for i in range(len(trees)):
    clf = RandomForestClassifier(n_estimators = int(trees[i]))
    clf.fit(train_x,train_y.values.ravel())
    pred_y = clf.predict_proba(valid_x)
    pred_y_list.append(pred_y)


columns_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns = columns_list)

#print("Log-loss: ", log_loss(valid_y,pred_y))

for i in range(len(valid_y)):
    index = columns_list.index(valid_y.iloc[i].item())
    yij.iloc[i,index] = 1
Multi_class_log_loss_list = []
for i in range(len(pred_y_list)):
    Multi_class_log_loss_value = logloss(pred_y_list[i],yij)
    print("The Multi Class Log Loss for RandomForest is: ",Multi_class_log_loss_value , " Number of Trees: ", trees[i])
    Multi_class_log_loss_list.append(Multi_class_log_loss_value)

print("Accuracy ", clf.score(valid_x,valid_y))

fig = plt.figure()
fig.suptitle('Log Loss Tests', fontsize=20)
plt.xlabel('Number of Trees', fontsize=18)
plt.ylabel('Multi-Class Log Loss', fontsize=16)
plt.plot(trees, Multi_class_log_loss_list)
