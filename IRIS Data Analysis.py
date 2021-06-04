# -*- coding: utf-8 -*-
"""
Created on Sat May  8 02:09:07 2021

@author: Hamza
"""
import pandas as pd

# Splitting Data
from sklearn.model_selection import train_test_split

# Modeling
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate

path=("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data")
headernamers=["Sepal Length(cm)", "Sepal Width(cm)", "Petal Length(cm)", "Petal Width(cm)", "Class"]

dataset= pd.read_csv(path, names = headernamers)
# dataset.info()


'''KNN'''
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values
scaler = StandardScaler()
X = scaler.fit_transform(x) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score=classifier.score(X_train, y_train)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix KNN:")
print(result)
result1 = accuracy_score(y_test,y_pred)
print("Accuracy KNN:",result1)
print("Test score KNN: ",score)

'''Decision Tree'''

feature_cols =  ["Sepal Length(cm)", "Sepal Width(cm)", "Petal Length(cm)", "Petal Width(cm)"]
output_cols = ["Class"]
x = dataset[feature_cols] # Features
y = dataset[output_cols] # Target variable
X = scaler.fit_transform(x) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
clf = DecisionTreeClassifier(max_depth = 10, criterion = 'entropy')
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=classifier.score(X_train, y_train)

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix DT:")
print(result)
result1 = accuracy_score(y_test,y_pred)
print("Accuracy DT:",result1)
print("Test score DT: ",score)


'''Cross Validation KNN For Decision Tree'''

metrics=["precision_micro", "accuracy", "balanced_accuracy"]


'''KNN'''
import warnings
warnings.filterwarnings("ignore")


best_score=0
best_clf=None
for i in range (10):
    clf = KNeighborsClassifier(n_neighbors = i)
    scores= cross_validate(clf, X_train, y_train, cv =5, scoring = metrics)
    
    scores=scores["test_accuracy"]
    avgscore = sum(scores)/len(scores)
    
    if(scores > avgscore).all:
        best_score = avgscore
        best_clf = clf
        
best_clf.fit(X_train, y_train)

score= best_clf.score(X_test,y_test)
train_score= best_clf.score(X_train,y_train)
print("Test cross validation KNN: " , score)
print("Train cross validation KNN: " , train_score)

'''DT'''
import warnings
warnings.filterwarnings("ignore")


best_score=0
best_clf=None
for i in range (100):
    clf = DecisionTreeClassifier(max_depth = i, criterion = 'entropy')
    scores= cross_validate(clf, X_train, y_train, cv =5, scoring = metrics)
    
    scores=scores["test_accuracy"]
    avgscore = sum(scores)/len(scores)
    
    if(scores > avgscore).all:
        best_score = avgscore
        best_clf = clf
        
best_clf.fit(X_train, y_train)

score= best_clf.score(X_test,y_test)
train_score= best_clf.score(X_train,y_train)
print("Test cross validation DT: " , score)
print("Train cross validation DT: " , train_score)