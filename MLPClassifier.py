# Template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from datetime import datetime


# Multi-Class Log-Loss

def logloss(pred_y, y_ij):
    max_p = 1 - 1e-15
    min_p = 1e-15
    num_of_products = len(pred_y)

    pred_y[(pred_y >= max_p)] = max_p
    pred_y[(pred_y <= min_p)] = min_p

    log_prob = np.log(pred_y)
    y_multi_prob = np.multiply(y_ij, log_prob)
    y_multi_prob = np.array(y_multi_prob)
    sum_value = np.sum(y_multi_prob)
    log_loss_value = -np.divide(sum_value, num_of_products)
    return log_loss_value


np.random.seed(12)
# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:, 1:94]
data_y = data.iloc[:, 94:95]

# test data import
train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.3, random_state=40)


# TODO -----------
# max_iter_list, pred_y_list = np.concatenate((np.arange(5, 51, 5), np.arange(60, 301, 20))), []
max_iter_list, pred_y_list = np.arange(10, 30, 1), []

for mi in max_iter_list:
    mlp = MLPClassifier(max_iter=mi, shuffle=False)
    mlp.fit(train_x, train_y.values.ravel())
    pred_y_list.append(mlp.predict_proba(valid_x))
# ----------------

columns_list = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]

yij = pd.DataFrame(0, index=np.arange(len(valid_y)), columns=columns_list)
for i in range(len(valid_y)):
    index = columns_list.index(valid_y.iloc[i].item())
    yij.iloc[i, index] = 1


mcll_list = []
for i in range(len(pred_y_list)):
    Multi_class_log_loss = logloss(pred_y_list[i], yij)
    mcll_list.append(Multi_class_log_loss)
    print(f"The Multi Class Log Loss is {Multi_class_log_loss} when max_iter is {max_iter_list[i]}")

#
# fig = plt.figure()
# fig.suptitle('MLPClassifier', fontsize=20)
# plt.xlabel('Max_iter', fontsize=16)
# plt.ylabel('Multi-Class Log Loss', fontsize=16)
# plt.plot(max_iter_list, mcll_list)
# plt.show()

# run the best parameter
start_time = datetime.now()
mlpb = MLPClassifier(max_iter=22, shuffle=False)
mlpb.fit(train_x, train_y.values.ravel())
print(f'The multi_class_log_loss of MLPclassifier is {logloss(mlpb.predict_proba(valid_x), yij)} when max_iter is 22.')
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
