##Course: RAI-833 Deep Learning by Dr. Shahbaz Khan
#Spring 2026
# Tutorial 1
# Written by Tahniat Khayyam

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# print("hello")

# Code from Tutorial

# class Perceptron (object):
#     def __init__(self,eta=0.01,n_iter=10):
#         self.eta=eta #learning rate
#         self.n_iter=n_iter #iter
#         # self.w_=w_
#     def weighted_sum(self,X):
#         return np.dot(X,self.w[1:]+self.w_[0])
#     def predict (self,X):
#         return np.where(self.weighted_sum(X)>=0.0,1,-1) #could either be 1 (if abv 0)
    
#     def fit(self,X,y):
#         # init weights to 0
#         self.w_=np.zeros(1+X.shape[1])
#         self.errors_=[]
#         print("Weights: ",self.w_)
#         for _ in range(self.n_iter):
#             error=0
#             for xi,y in zip(X,y):
#                 y_pred=self.predict(xi) #wt curr input for each loop
#                 update=self.eta*(y-y_pred)
#                 # w=wi+update
#                 # update=n*(y-yi)*X
#                 self.w_[1:]=self.w_[1:]+ update*xi
#                 print("updated weights: ",self.w_[1:])
#                 # update the bias too
#                 self.w_[0]=self.w_[0]+update
#                 error+=int(update!=0.0) #increment if model pred incorrect, so if update is not 0, 

#             self.errors_.append(error) #store errors 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df=shuffle(df)
print(df.head())
#separating the data
X=df.iloc[:,0:4].values
y=df.iloc[:,4].values
print(X[0:5])
print(y[0:5])

from sklearn.model_selection import train_test_split

#splitting the data
train_data,test_data,train_labels,test_labels=train_test_split(X,y,test_size=0.25)

#encoding 1 (setosa) ,-1
train_labels=np.where(train_labels=='Iris-setosa',1,-1)
test_labels=np.where(test_labels=='Iris-setosa',1,-1)


print("train data: ",train_data[0:2])
print("train labels: ",train_labels[0:2])

print("test data: ",test_data[0:2])
print("test labels: ",test_labels[0:2])
from sklearn.linear_model import Perceptron
# let's train the perceptron
perceptron=Perceptron(eta0=0.1,max_iter=10)
# fit
perceptron.fit(train_data,train_labels)

# making predictions
test_preds=perceptron.predict(test_data)
print("test preds: ",test_preds) # print array of pred labels
print(test_data.shape)



# ----------------------------------#
# Task 1
# ----------------------------------#

# taking in manual input 
man_input=input("Enter four features: sepal length, sepal width, petal length, petal width. Take care of commas \n")
# input = 4,5,3.2,1
 
man_array = np.array(list(map(float, man_input.split(',')))).reshape(1, -1)

man_pred=perceptron.predict(man_array)
print(man_pred)


# ----------------------------------#
# Task 2
# ----------------------------------#

print("\n\n\n Attempting Task 2")
print("Task2.1 Adding sigmoid function...")

def sigmoid(z):
        return 1 / (1 + np.exp(-z))
class Perceptron2 (object):
    def __init__(self,eta=0.01,n_iter=10):
        self.eta=eta #learning rate
        self.n_iter=n_iter #iter
        # self.w_=w_
    # def weighted_sum(self,X):
    #     return np.dot(X,self.w[1:]+self.w_[0])
    
    def predict (self,X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return (sigmoid(z) >= 0.5).astype(int)
    
    def fit(self,X,y):
        # init weights to 0
        self.w_=np.zeros(1+X.shape[1])
        self.errors_=[]
        print("Weights: ",self.w_)
        for _ in range(self.n_iter):
            error=0
            for xi,yi in zip(X,y):
                y_pred=self.predict(xi) #wt curr input for each loop
                update=self.eta*(yi-y_pred)
                # w=wi+update
                # update=n*(y-yi)*X
                self.w_[1:]=self.w_[1:]+ update*xi
                # print("updated weights: ",self.w_[1:])
                # update the bias too
                self.w_[0]=self.w_[0]+update
                error+=int(update!=0.0) #increment if model pred incorrect, so if update is not 0, 
            print("final weights: ",self.w_[1:])
            self.errors_.append(error) #store errors 

#splitting the data
train_data,test_data,train_labels,test_labels=train_test_split(X,y,test_size=0.25)

#encoding 1 (setosa) ,-1
print("\n\n\nTask2.1 Replacing labels...")
train_labels=np.where(train_labels=='Iris-setosa',1,0)
test_labels=np.where(test_labels=='Iris-setosa',1,0)


print("train data: ",train_data[0:2])
print("train labels: ",train_labels[0:2])

print("test data: ",test_data[0:2])
print("test labels: ",test_labels[0:2])


# let's train the perceptron
perceptron2=Perceptron2(eta=0.1,n_iter=10)
# fit
perceptron2.fit(train_data,train_labels)

# making predictions
print("making predictions again...\n")
test_preds=perceptron2.predict(test_data)
print("test preds: ",test_preds) # print array of pred labels
print(test_data.shape)
# 