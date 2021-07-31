import xgboost as xg
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

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

def y_generate():
    columns_list = ["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"]
    yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns = columns_list)
    for i in range(len(valid_y)):
        index = columns_list.index(valid_y.iloc[i].item())
        yij.iloc[i,index] = 1
    return yij

def classifier_(clf):
    clf.fit(train_x,train_y.values.ravel())
    pred_value = clf.predict_proba(valid_x)
    log_loss_clf = logloss(pred_value,yij)
    score_xgb = clf.score(valid_x,valid_y)
    return log_loss_clf, score_xgb


if __name__ == "__main__":
    np.random.seed(12)
    # train data import
    data = pd.read_csv('train.csv')
    data_x = data.iloc[:, 1:94]
    data_y = data.iloc[:, 94:95]
    # test data import
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40)
    skb = SelectKBest(mutual_info_classif, k=85)
    skb.fit(train_x, train_y.values.ravel())
    train_x = skb.transform(train_x)
    valid_x = skb.transform(valid_x)

    yij = y_generate()
    xgb = xg.XGBClassifier(learning_rate =0.1, n_estimators=450,max_depth=7,min_child_weight=4,
                    objective='multi:softprob', eval_metric='mlogloss')
    mlpb = MLPClassifier(solver='adam', max_iter=45, shuffle=True, activation='logistic', random_state=0)
    clf=SVC(C = 1.0, kernel='rbf',probability=True)
    rd = RandomForestClassifier(n_estimators = 178, criterion = 'gini', max_depth = 35)
    # using ensemble method named "soft voting classifier"

    eclf = VotingClassifier(estimators=[('xgb',xgb),('mlp',mlpb),('svc',clf),('rf',rd)],voting='soft')
    eclf = eclf.fit(train_x,train_y.values.ravel())
    pred_y = eclf.predict(valid_x)
    score_1 = eclf.score(valid_x, valid_y)
    pred_weight = eclf.predict_proba(valid_x)
    log_loss_1 = logloss(pred_weight, yij)
    print("log loss of soft: ",log_loss_1)
    print("score : ", score_1)

    log_loss_xg, score_xg = classifier_(xgb)
    log_loss_mlpb, score_mlpb = classifier_(mlpb)
    log_loss_rd, score_rd = classifier_(rd)
    log_loss_svc, score_svc = classifier_(clf)

    x_list = ['Ensemble', 'XGBoost', 'MLP', 'RandomForest', 'SVC']
    log_loss_list = []
    log_loss_list.append(log_loss_1)
    log_loss_list.append(log_loss_xg)
    log_loss_list.append(log_loss_mlpb)
    log_loss_list.append(log_loss_rd)
    log_loss_list.append(log_loss_svc)
    score_list = []
    score_list.append(score_1)
    score_list.append(score_xg)
    score_list.append(score_mlpb)
    score_list.append(score_rd)
    score_list.append(score_svc)

    plt.ylim(0,1.1)
    plt.bar(x = np.arange(len(x_list)),height = log_loss_list,label = 'log loss', color = 'b', width = 0.4)
    plt.bar(x = np.arange(len(x_list))+0.4,height = score_list,label = 'accuarcy', color = 'r', width = 0.4)
    plt.xticks(np.arange(len(x_list))+0.2, x_list)
    for x,y in zip(np.arange(len(x_list)),log_loss_list):
        #print(x)
        plt.text(x-0.2,y+0.02,'%.2f' %y)
    for x,y in zip(np.arange(len(x_list)),score_list):
        #print(x)
        plt.text(x+0.2,y+0.02,'%.2f' %y)
    plt.title('Log Loss and Accuarcy of each model')
    plt.legend()
    plt.show()