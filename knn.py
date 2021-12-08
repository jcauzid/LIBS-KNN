#/*##########################################################################
# Copyright (C) 2020-2021 The University of Lorraine - France
#
# This file is part of the LIBS-KNN toolkit developed at the GeoRessources
# Laboratory of the University of Lorraine, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os  
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

Ninerals = [
    'Wolframite', 'Tourmaline', 'Sphalerite','Rutile','Quartz', 'Pyrite', 'Orthose', 'Muscovite', 'Molybdenite', 'Ilmenite',
    'Hematite', 'Fluorite', 'Chlorite', 'Chalcopyrite', 'Cassiterite', 'Biotite', 'Arsenopyrite', 'Apatite', 'Albite']
Elemnts = ['Si', 'Al', 'K', 'Na', 'Ca', 'Fe', 'Mg', 'Mn', 'CaF', 'S', 'Ti', 'Sn',
       'W', 'Cu', 'Zn', 'Ba', 'Rb', 'Sr', 'Li', 'As', 'Mo']

def get_res(list_res):
    max_value = None
    max_idx = None
    for idx, num in enumerate(list_res):
        if (max_value is None or num > max_value):
            max_value = num
            max_idx = idx
    return [max_idx, max_value] 


#### import Data :
# Input
Data = pd.read_csv("Train_Test.csv")
X = Data[Elemnts]
INPUTS = X.to_numpy()
# Output :
y = Data["Target"].to_frame()
OUPUTS = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split( INPUTS, OUPUTS, test_size=0.25, random_state=42)


#### Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = 1).fit(X_train,y_train.ravel())
# Validation 
DF_Validation = pd.read_csv("Validation.csv" )
#DF_Validation = DF_Validation.iloc[0:2000]
Input_Set_Validation = DF_Validation[Elemnts]
INPUTS_Validation = Input_Set_Validation.to_numpy()


# saving result :
R_Minerals = []
R_Probability = []
for k in range(len(DF_Validation)):
    yhat_ = neigh.predict(INPUTS_Validation[k:k+1])
    yhat_proba = neigh.predict_proba(INPUTS_Validation[k:k+1])
    [max_idx, max_value] = get_res(list_res = yhat_proba[0] )
    R_Minerals.append(Ninerals[max_idx])
    R_Probability.append(max_value)
Minls = {'Minirals': R_Minerals, 'Prob': R_Probability}
df = pd.DataFrame(data=Minls)

df.to_csv("Results_KNN_{}_samples.csv".format(len(DF_Validation)))


"""
Ks = 40
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    print("============================= case k = ",n)
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train.ravel())
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
"""
