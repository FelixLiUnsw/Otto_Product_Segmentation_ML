# Template
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from datetime import datetime
start_time = datetime.now()

np.random.seed(12)

np.random.seed(12)
# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:, 1:94]
data_y = data.iloc[:, 94:95]

# test data import
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40)

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
alpha_list = np.linspace(0.0001, 3.0, 30)
accuracy_true_list,mul_logloss_true_list=[],[]
for a in alpha_list:
    clf=MultinomialNB(alpha =a, fit_prior = True , class_prior = None)
    clf.fit(train_x,train_y.values.ravel())
    pred_y=clf.predict_proba(valid_x)
    pred_y2=clf.predict(valid_x)
    Multi_class_log_loss = logloss(pred_y, yij)
    mul_logloss_true_list.append(Multi_class_log_loss)
    accuracy_true_list.append(accuracy_score(valid_y, pred_y2))
    #print("The Multi Class Log Loss: ", Multi_class_log_loss)
    #print("MultinomialNB accuracy: ", accuracy_score(valid_y, pred_y2))

accuracy_false_list,mul_logloss_false_list=[],[]
for a in alpha_list:
    clf=MultinomialNB(alpha =a, fit_prior = False , class_prior = None)
    clf.fit(train_x,train_y.values.ravel())
    pred_y=clf.predict_proba(valid_x)
    pred_y2=clf.predict(valid_x)
    Multi_class_log_loss = logloss(pred_y, yij)
    mul_logloss_false_list.append(Multi_class_log_loss)
    accuracy_false_list.append(accuracy_score(valid_y, pred_y2))
    #print("The Multi Class Log Loss: ", Multi_class_log_loss)
    #print("MultinomialNB accuracy: ", accuracy_score(valid_y, pred_y2))


#multi logloss plot with true and false
def logloss_plot(mul_logloss_true_list,mul_logloss_false_list):
    plt.title('MultinomialNB')
    plt.plot(alpha_list,mul_logloss_true_list,c='green')
    plt.plot(alpha_list,mul_logloss_false_list,c='blue')
    plt.legend(labels=['fit_prior=true','fit_prior=false'])
    #plt.yticks(range(0,4,1))
    plt.xlabel('alpha')
    plt.ylabel('multi logloss')
    plt.show()

#accuracy plot with true and false
def accuracy_plt(accuracy_true_list,accuracy_false_list):
    plt.title('MultinomialNB')
    plt.plot(alpha_list,accuracy_true_list,c='green')
    plt.plot(alpha_list,accuracy_false_list,c='blue')
    plt.legend(labels=['fit_prior=true','fit_prior=false'])
    #plt.yticks(range(0,4,1))
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.show()

#logloss_plot(mul_logloss_true_list,mul_logloss_false_list)
#accuracy_plt(accuracy_true_list,accuracy_false_list)

#multi logloss plot with true
def logloss_plot_true(mul_logloss_true_list):
    plt.title('MultinomialNB')
    plt.plot(alpha_list, mul_logloss_true_list, c='green')
    plt.xlabel('alpha')
    plt.ylabel('multi logloss')
    plt.tight_layout()
    plt.show()

#accuracy plot with true
def accuracy_plot_true(accuracy_true_list):
    plt.title('MultinomialNB')
    plt.plot(alpha_list, accuracy_true_list, c='green')
    plt.xlabel('alpha')
    plt.ylabel('accuracy')

    plt.tight_layout()
    plt.show()

logloss_plot_true(mul_logloss_true_list)
accuracy_plot_true(accuracy_true_list)

max_acc=max(accuracy_true_list)
for index in range(len(accuracy_true_list)):
    if accuracy_true_list[index]==max_acc:
        print("alpha:",index/10," ,best accuracy:",max_acc," ,best logloss:",mul_logloss_true_list[index])

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))