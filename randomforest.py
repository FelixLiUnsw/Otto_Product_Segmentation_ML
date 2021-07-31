# Ensemble Method
# Random Forest Algorithm
# Jiaxuan Li z5086369
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# multi-class log loss
# The original formula has been shown in the evaluation page in otto kaggle competition
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

# Randomforest classifier
# find the predict value with different n_estimators (trees) in a range from 25 - 300, increment 25 each
# return a list --> the list of predict y value
def classifiers(depth):
    trees = np.linspace(25, 300, 12)
    pred_y_list = []
    for i in range(len(trees)):
        clf = RandomForestClassifier(n_estimators = int(trees[i]), criterion = 'gini', max_depth = depth)
        clf.fit(train_x,train_y.values.ravel())
        pred_y = clf.predict_proba(valid_x)
        pred_y_list.append(pred_y)
    return pred_y_list

def y_generate():
    columns_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns = columns_list)
    for i in range(len(valid_y)):
        index = columns_list.index(valid_y.iloc[i].item())
        yij.iloc[i,index] = 1
    return yij


def ploting(pred_y_matrix, yij_matrix, depth_value,sentence):
    Multi_class_log_loss = []
    accuracy = []
    trees = np.linspace(25, 300, 12)
    for i in range(len(pred_y_matrix)):
        # log loss
        MCLL = logloss(pred_y_matrix[i],yij_matrix)
        print("The Multi Class Log Loss for ",sentence," is: ", MCLL, " Number of Trees: ", trees[i], " Depth = ", depth_value)
        Multi_class_log_loss.append(MCLL)
        # accuarcy
    fig = plt.figure()
    if(depth_value == 10):
        title_ = 'depth = 10'
    elif(depth_value == 20):
        title_ = 'depth - 20'
    elif(depth_value == 30):
        title_ = 'depth - 30'
    elif(depth_value == 40):
        title_ = 'depth - 40'
    fig.suptitle(title_, fontsize=20)
    plt.xlabel('Number of Trees', fontsize=18)
    plt.ylabel('Multi-Class Log Loss', fontsize=16)
    plt.plot(trees, Multi_class_log_loss)


def cross_validation(lower,upper,depth_lower,depth_upper):
    trees = np.linspace(lower,upper,4)
    trees = trees.astype(int).tolist()
    depth_ = np.linspace(depth_lower,depth_upper,3)
    depth_ = depth_.astype(int).tolist()
    parameter_space = {
        'n_estimators': trees,
        'criterion': ["gini"],
        'max_depth': depth_,
    }
    clf = RandomForestClassifier() 
    kfold = KFold(n_splits = 10)
    grid = HalvingGridSearchCV(clf, parameter_space, cv=kfold)
    grid.fit(train_x,train_y.values.ravel())

    return grid.best_params_

if __name__ == "__main__":
    np.random.seed(12)
    # data import for inputs and outputs
    data = pd.read_csv('train.csv')
    data_x = data.iloc[:,1:94]
    data_y = data.iloc[:,94:95]
    train_x, valid_x, train_y, valid_y=train_test_split(data_x,data_y,test_size=0.3,random_state=40)
    y_ij = y_generate()
    print("Start Ploting")
    # Random Forest
    print("Random Forest")
    pred_y_10_r = classifiers(10)
    ploting(pred_y_10_r, y_ij,10,"randomforest")
    pred_y_20_r = classifiers(20)
    ploting(pred_y_20_r, y_ij,20,"randomforest")
    pred_y_30_r = classifiers(30)
    ploting(pred_y_30_r, y_ij,30,"randomforest")
    pred_y_40_r = classifiers(40)
    ploting(pred_y_40_r, y_ij,40,"randomforest")

    # best random forest parameter in range 160-185 trees and 25-35 depth
    # using cross validation to find the local minimum (HalvingGridSearchCV)
    best_rf= cross_validation(165,185,25,35)
    print("Optimized random forest: ", best_rf)
    trees = best_rf.get('n_estimators')
    depth = best_rf.get('max_depth')
    start_time = datetime.now()
    clf = RandomForestClassifier(n_estimators = trees, criterion = 'gini', max_depth = depth)
    clf.fit(train_x,train_y.values.ravel())
    pred_y = clf.predict_proba(valid_x)
    end_time = datetime.now()
    y_ij = y_generate()
    # print the best values' log loss , accuarcy, and time duration
    best_value = logloss(pred_y, y_ij)
    print("The log loss is: ", best_value)
    print('Duration: {}'.format(end_time - start_time))
    pred = clf.predict(valid_x)
    score = accuracy_score(valid_y,pred)
    print("accuracy is: ",score)
    plt.show()
