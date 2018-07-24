# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 19:23:22 2018

@author: Pardhu Gopalam
"""

#imports
# pandas for dataframe & sklearn for metrics and classifiers
# time for considering the 
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from time import time
from sklearn.metrics import classification_report

#defining accuracy function
def accuracy(actual_data,predict_output):
    a = 0
    A = len(actual_data)
    for i in range(A):
        if actual_data[i] == predict_output[i]:
            a = a+1
    x = (a/A)*100
    return(x) 

#importing data into pandas dataframe    
shares_list= []
datafile_name = 'OnlineNewsPopularity.csv'
data =pd.read_csv(datafile_name)
#prices_df = total_df['Close']
shares_df = data[' shares']
threshold = 1400
shares_bin = shares_df

#binarising data as popular and not populra with 1400 as threshold. 
data.loc[shares_bin <= 1400,' shares'] = 0 # not popular
data.loc[shares_bin > 1400	,' shares'] = 1  #popular
data_considered = data

#deleting non predictor attributes
del data_considered['url']
del data_considered[' timedelta']
X_train, X_test, y_train, y_test = train_test_split(data, shares_bin, test_size=0.7)

t1=time()
gnb = GaussianNB()
gnb.fit(X_train,y_train)
GNB_predict_output = gnb.predict(X_test)
GNB_actual_data = np.asarray(X_test[' shares'])
t2=time()

#recall & precision functions using metrics
GNB_Acc = accuracy(GNB_actual_data,GNB_predict_output)    
GNB_rec = recall_score(GNB_actual_data, GNB_predict_output)
GNB_preci = precision_score(GNB_actual_data, GNB_predict_output)
#average_precision = average_precision_score(GNB_actual_data, GNB_predict_output)


print('--------Gaussian Naive Bayes-------------')
print('Gaussian Naive Bayes accuracy:', GNB_Acc,'%')
print('Gaussian Naive Bayes recall:', GNB_rec)
print('Gaussian Naive Bayes precision:', GNB_preci)
print ("time take by Gaussian naive bayes: ", t2-t1)
)

# classification report
print('---------------------------------------')
print(classification_report(y_test,GNB_predict_output))
