'''
Name: Andrei Oprea
u796299
'''


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


to2d = lambda img: np.reshape(img, (int(sqrt(len(img))), int(sqrt(len(img)))))

data = np.load('2036924_face.npz')
X_train = data['X_train']
y_train = data['y_train']
X_valid = data['X_valid']
y_valid = data['y_valid']
X_test = data['X_test']
y_test = data['y_test']

X_trainval = np.concatenate([X_train,X_valid])
y_trainval = np.concatenate([y_train,y_valid])

print(data.files)

#1.2.1
fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=len(np.unique(y_train)))

index = []
for i in np.unique(y_train):
    for labn in range(len(y_train)):
        if y_train[labn] == i:
            index.append(labn)
    ax[i].imshow(to2d(X_train[index[i]]), cmap='gray')
    ax[i].set_title(i)
    index = []
plt.show()
plt.clf()

# 1.2.2
y_total = np.concatenate([y_train, y_valid, y_test])
plt.hist(y_total, bins=len(np.unique(y_total)))
plt.suptitle('Histogram of all target labels', fontsize = 15)
plt.title('Dataset is not balanced: 0 has most, then 2 and then 1')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.xticks(np.unique(y_total))
plt.show()
plt.clf()


#1.3.1
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
fig, ax2 = plt.subplots(figsize=(10, 7), nrows=2, ncols=len(np.unique(y_test)))

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

org = y_test
pred = clf.predict(X_test)
index = []
for i in np.unique(y_test):
    for labn in range(len(y_test)):
        if y_test[labn] == i and org[labn] != pred[labn]:
            index.append(labn)
    for j in range(2):
        ax2[j,i].imshow(to2d(X_test[index[j]]), cmap='gray')
        ax2[j,i].set_title(f'{org[index[j]]} missclassified as {pred[index[j]]}')
    index = []
plt.show()
plt.clf()
print(f"As a second metric to evaluate the classifier, the F1 score is: { metrics.f1_score(org, pred, average='micro') }")

#1.3.2
from sklearn.neighbors import KNeighborsClassifier
best_score = 0
#After running it with k in range(1, len(y_train)), saw that after k=500 it stays at the same accuracy score of 0.54
# So for effieciency ill run it till k=100
for k in range(1, 100, 2):
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(X_train, y_train)
    score = clf.score(X_valid, y_valid)
    if score > best_score:
        best_score = score
        best_parameters = {'n_neighbors':k}

knn = KNeighborsClassifier(**best_parameters).fit(X_trainval,y_trainval)
test_score = knn.score(X_test,y_test)

print(f'Best score on validation set: {best_score}')
print(f'Best parameters: {best_parameters}')
print(f'Test set score with best parameters: {test_score}')

#1.3.3
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from itertools import product
from imblearn.over_sampling import SMOTE
import time

def Balancing(X_train, y_train):
    X_train_bl, y_train_bl =SMOTE(random_state=0, n_jobs=-1).fit_resample(X_train,y_train)
    return X_train_bl, y_train_bl

def DataScaling(X_train, X_valid, X_test, X_trainval):
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    X_trainval_scaled = scaler.transform(X_trainval)
    return X_train_scaled, X_valid_scaled, X_test_scaled, X_trainval_scaled


NMF_params = {'n_components': 150, 'init': 'nndsvda', 'max_iter': 1000, 'tol': 0.01, 'random_state': 0}
PCA_params = {'n_components': 0.99, 'random_state': 0}

def DimReduction(type, params):
    X_train_scaled, X_valid_scaled, X_test_scaled, X_trainval_scaled = DataScaling(X_train, X_valid, X_test, X_trainval) #Scaling the data before dimReduction

    type = type(**params).fit(X_train_scaled)
    X_train_dr = type.transform(X_train_scaled)
    X_valid_dr = type.transform(X_valid_scaled)
    X_test_dr = type.transform(X_test_scaled)
    X_trainval_dr = type.transform(X_trainval_scaled)
    print(f'\n>>>Using the {type} technique, features were reduced from {X_train_scaled.shape} to {X_train_dr.shape} for the training set')
    return X_train_dr, X_valid_dr, X_test_dr, X_trainval_dr


def optimizer(model, parameters):
    best_score = 0
    param_combs = [dict(zip(parameters,values)) for values in product(*parameters.values())] #defining all parameter combinations
    for i in param_combs:
        clf = model(**i).fit(X_train_use, y_train)  #fitting given model on all parameter combinations
        score = clf.score(X_valid_use, y_valid)
        if score > best_score:  #choosing best parameters
            best_score = score
            best_parameters = i #remambering the index of the best parameters

    t0 = time.time()
    optimized_clf = model(**best_parameters).fit(X_trainval_use,y_trainval)
    test_score = optimized_clf.score(X_test_use,y_test)
    t1 = time.time()    #testing and timing the optimized model on the testing set
    run_time = t1 - t0

    org = y_test
    pred = optimized_clf.predict(X_test_use)

    print(f'Best parameters: {best_parameters}')
    print(f'Test set accuracy score with best parameters: {test_score}')
    print(f"Test set F1 score with best parameters: {f1_score(org, pred, average='weighted')}")
    print(f'Optimized model run-time: {run_time}')

#Search-space of hyperparameters
svc_params = {'kernel': ['rbf'],
              'random_state': [0],
              'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10, 'auto', 'scale']
              }
tree_params = {'max_depth': list(range(1,50)),
               'random_state': [0]
               }
knn_params = {'n_neighbors': list(range(1,150,2)),
              'weights': ['distance'],
              'n_jobs': [-1]
              }
mlp_params = {'solver': ['adam'],
              'hidden_layer_sizes': [[10,10,10], [10,10,10,10], [50,50], [50,50,50], [50,50,50,50], [100], [100,100], [100,100,100], [100,100,100,100]],
              'max_iter': [1000],
              'early_stopping': [True],
              'random_state': [0],
              }
rfc_params = {'n_estimators': [100,500,1000, 5000],
              'criterion': ['gini', 'entropy'],
              'max_depth': [2,4,6,8,10,None],
              'n_jobs': [-1],
              'random_state': [0]
              }
vc_params = {'estimators': [[('nn', MLPClassifier(solver='adam', hidden_layer_sizes=[100], max_iter=1000, early_stopping=True, random_state=0)), ('svc', SVC(kernel='rbf', random_state=0, C=10, gamma='scale'))], [('knn', KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=-1)),('tree', DecisionTreeClassifier(max_depth=10, random_state=0))]],
             'n_jobs': [-1]
             }

# # #Just scaling --- Takes too long and seams like is not needed anymore if not for comparison
# X_train, y_train = Balancing(X_train, y_train)
# X_train_use, X_valid_use, X_test_use, X_trainval_use = DataScaling(X_train, X_valid, X_test, X_trainval)
# for model, parameters in zip([SVC, DecisionTreeClassifier, KNeighborsClassifier, MLPClassifier, RandomForestClassifier],[svc_params, tree_params, knn_params, mlp_params, rfc_params]):
#     print(f'\n------> Results for {model}:')
#     optimizer(model, parameters)


X_train, y_train = Balancing(X_train, y_train)
#PCA + scaling
X_train_use, X_valid_use, X_test_use, X_trainval_use = DimReduction(PCA, PCA_params)
for model, parameters in zip([SVC, DecisionTreeClassifier, KNeighborsClassifier, MLPClassifier, RandomForestClassifier],[svc_params, tree_params, knn_params, mlp_params, rfc_params]):
    print(f'\n------> Results for {model}:')
    optimizer(model, parameters)

#MNF + scaling
X_train_use, X_valid_use, X_test_use, X_trainval_use = DimReduction(NMF, NMF_params)
for model, parameters in zip([SVC, DecisionTreeClassifier, KNeighborsClassifier, MLPClassifier, RandomForestClassifier],[svc_params, tree_params, knn_params, mlp_params, rfc_params]):
    print(f'\n------> Results for {model}:')
    optimizer(model, parameters)

#Running 2 combinations of optimized algorithms in the voting classifier same way as above (principle of optimizer function)
vc_param_combs = [dict(zip(vc_params,values)) for values in product(*vc_params.values())]
for params in vc_param_combs:
    print(f'\n------> Results for {VotingClassifier}:')
    t0 = time.time()
    clf = VotingClassifier(**params).fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    t1 = time.time()
    run_time = t1 - t0
    print(f'Optimized algorithms used: {params}')
    print(f'Test set accuracy score with best parameters: {test_score}')
    org = y_test
    pred = clf.predict(X_test)
    print(f"Test set F1 score with best parameters: {f1_score(org, pred, average='weighted')}")
    print(f'Shortest run-time: {run_time}')