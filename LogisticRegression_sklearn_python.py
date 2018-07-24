# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 19:25:40 2018

@author: Pardhu Gopalam
"""

#imports
# pandas for dataframe & sklearn for metrics and classifiers
# time for considering the 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


t3 = time()
LR = LogisticRegression()
LR.fit(X_train,y_train)
LR_predict_output = LR.predict(X_test)
LR_actual_data = np.asarray(X_test[' shares'])
t4=time()

#recall & precision functions using metrics
LR_Acc = accuracy(LR_actual_data,LR_predict_output)    
LR_rec = recall_score(LR_actual_data, LR_predict_output)
LR_preci = precision_score(LR_actual_data, LR_predict_output)
#average_precision = average_precision_score(LR_actual_data, LR_predict_output)

print('--------Logstic regression-------------')
print('Logstic regression accuracy:',LR_Acc,'%')
print('logistic regression recall:',LR_rec)
print('logistic regression precision:', LR_preci)
print ("time take by Logistic regression: ", t4-t3)

# classification report
print('---------------------------------------')
print(classification_report(y_test,LR_predict_output))