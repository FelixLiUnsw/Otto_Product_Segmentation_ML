# Random Forest Classification
# Jiaxuan Li
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

np.random.seed(12)
# train data import
data = pd.read_csv('train.csv')
data_x = data.iloc[:,1:94]
data_y = data.iloc[:,94:95]

#feature_list = list(data_x.columns)
feature_list = np.linspace(0,93,94).tolist()
counter_array = np.zeros(93)

for i in range(len(data_x)):
    for j in range(len(counter_array)):
        if data_x.iloc[i,j] != 0:
            counter_array[j]+= 1
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
        #print(i)
    else:
        plt.bar(feature_list[i],height = counter_array[i] ,color = 'r')
        print("Abandoned Feature：Feature", i+1)
        abandoned_feature_list.append(i)
plt.show()
