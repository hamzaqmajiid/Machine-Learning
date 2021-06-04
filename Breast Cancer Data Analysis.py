import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# For Splitting Data
from sklearn.model_selection import train_test_split

# For Modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

path=("archive/data.csv")

cancer= pd.read_csv(path)

cancer.drop(columns=['id', 'Unnamed: 32'], inplace = True)

cancer.isna().sum()/len(cancer.index)*100



# If the cancer is Benign, it will be 0
# If the cancer is Malignant, it will be 1

cancer['diagnosis'] = np.where(cancer['diagnosis'] == 'M', 1, 0)

cancer['diagnosis'].value_counts()/cancer.shape[0]*100


X = cancer.drop('diagnosis', axis = 1)
y = cancer['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3, random_state = 3030)

##### KNN #####

k = range(1,50,1)
testing_accuracy = []
training_accuracy = []
score = 0

for i in k:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    
    y_predict_train = knn.predict(X_train)
    training_accuracy.append(accuracy_score(y_train, y_predict_train))
    
    y_predict_test = knn.predict(X_test)
    acc_score = accuracy_score(y_test,y_predict_test)
    testing_accuracy.append(acc_score)
    
    if score < acc_score:
        score = acc_score
        best_k = i
        

print('This is the best K for KNeighbors Classifier: ', best_k, '\nAccuracy score is: ', score)

result = confusion_matrix(y_test, y_predict_test)
print("Confusion Matrix:","\n", result)

##### Decision Tree #####

depth = range(1,25)
testing_accuracy = []
training_accuracy = []
score = 0

for i in depth:
    tree = DecisionTreeClassifier(max_depth = i, criterion = 'entropy')
    tree.fit(X_train, y_train)
    
    y_predict_train = tree.predict(X_train)
    training_accuracy.append(accuracy_score(y_train, y_predict_train))
    
    y_predict_test = tree.predict(X_test)
    acc_score = accuracy_score(y_test,y_predict_test)
    testing_accuracy.append(acc_score)
    
    if score < acc_score:
        score = acc_score
        best_depth = i
        

print('This is the best depth for Decision Tree Classifier: ', best_depth, '\nAccuracy score is: ', score)

result = confusion_matrix(y_test, y_predict_test)
print("Confusion Matrix:","\n", result)

knn = KNeighborsClassifier(n_neighbors = 3)
tree = DecisionTreeClassifier(max_depth = 3, random_state = 3030)

def model_evaluation(model, metric):
    model_cv = cross_val_score(model, X_train, y_train, cv = StratifiedKFold(n_splits = 5), scoring = metric)
    return model_cv

knn_cv = model_evaluation(knn, 'recall')
tree_cv = model_evaluation(tree, 'recall')

for model in [knn, tree]:
    model.fit(X_train, y_train)

score_cv = [knn_cv.round(5), tree_cv.round(5)]
score_mean = [knn_cv.mean(), tree_cv.mean()]
score_std = [knn_cv.std(), tree_cv.std()]
score_recall_score = [recall_score(y_test, knn.predict(X_test)), 
            recall_score(y_test, tree.predict(X_test))]
method_name = [ 'KNN Classifier', 'Decision Tree Classifier']
cv_summary = pd.DataFrame({
    'method': method_name,
    'cv score': score_cv,
    'mean score': score_mean,
    'std score': score_std,
    'recall score': score_recall_score
})
print(cv_summary)
