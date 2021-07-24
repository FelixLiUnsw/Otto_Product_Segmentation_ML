# Random Forest Classification
# Jiaxuan Li z5086369
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# find unimportant features and save it into a list
# show unimportant features in a plot
# red bars are unimportant features
# blue bars are important features
def data_processing():
    counter_array = np.zeros(93)
    for i in range(len(data_x)):
        for j in range(len(counter_array)):
            if data_x.iloc[i,j] != 0:
                counter_array[j]+= 1

    feature_list = np.linspace(0,93,94).tolist()
    plt.figure(figsize=(30, 18))
    feature_list = [int(item) for item in feature_list]
    abandoned_feature_list=[]
    for i in range(len(feature_list)-1):
        if(counter_array[i] >= 5000):
            #print("here")
            plt.bar(feature_list[i],height = counter_array[i],color = 'b')
            print(i)
        else:
            plt.bar(feature_list[i],height = counter_array[i] ,color = 'r')
            print("Abandoned Feature：Feature", i+1)
            abandoned_feature_list.append(i)
    #plt.show()
    return abandoned_feature_list

# drop unimportant features from the features dataframe
def drop_unimportant_features(abandoned_feature_list):
    column_name = []
    for i in range(len(abandoned_feature_list)-1):
        column_name.append(data_x.columns[abandoned_feature_list[i]])
    data_x.drop(column_name, axis=1,inplace=True)
    return

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
    print(log_loss_value)
    return log_loss_value

# Randomforest classifier
# find the predict value with different n_estimators (trees) in a range from 25 - 300, increment 25 each
# return a list --> the list of predict y value
def randomforest(depth):
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


def ploting(pred_y_matrix, yij_matrix, depth_value):
    Multi_class_log_loss = []
    trees = np.linspace(25, 300, 12)
    for i in range(len(pred_y_matrix)):
        MCLL = logloss(pred_y_matrix[i],yij_matrix)
        print("The Multi Class Log Loss for RandomForest is: ", MCLL, " Number of Trees: ", trees[i], " Depth = ", depth_value)
        Multi_class_log_loss.append(MCLL)
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



if __name__ == "__main__":

    np.random.seed(12)
    # data import for inputs and outputs
    data = pd.read_csv('train.csv')
    data_x = data.iloc[:,1:94]
    data_y = data.iloc[:,94:95]
    #Data processing --> find and drop unimportant features
    abandoned_feature_list = data_processing()
    drop_unimportant_features(abandoned_feature_list)
    #split the data into train_x，valid_x, train_y, valid_y  --> saving as panda dataFrame
    train_x, valid_x, train_y, valid_y=train_test_split(data_x,data_y,test_size=0.3,random_state=40)

    #try different depth from 10-40


    # y_ij is the 2D matrix, it follows the required format for calcuating the log_loss
    # if it is class9, then y_ij = [0,0,0,0,0,0,0,0,1]
    y_ij = y_generate()
    print("Start Ploting")
    pred_y_10 = randomforest(10)
    pred_y_20 = randomforest(20)
    pred_y_30 = randomforest(30)
    pred_y_40 = randomforest(40)
    ploting(pred_y_10, y_ij,10)
    ploting(pred_y_20, y_ij,20)
    ploting(pred_y_30, y_ij,30)
    ploting(pred_y_40, y_ij,40)
    plt.show()