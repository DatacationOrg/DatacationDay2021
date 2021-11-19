import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data = pd.read_csv("train_brain.csv")

X = pd.DataFrame(data.iloc[:,:-1])
y = pd.DataFrame(data.iloc[:,-1])

cat_data = X.select_dtypes(include = "object")
num_data = X.select_dtypes(exclude = "object")

cat_data = SimpleImputer(strategy = "most_frequent").fit(cat_data).transform(cat_data)
cat_data = pd.DataFrame(cat_data)


for column in cat_data.keys():
    cat_data[column] = pd.factorize(cat_data[column])[0]

cat_data = StandardScaler(with_mean = False).fit(cat_data).transform(cat_data)
cat_data = pd.DataFrame(cat_data)



num_data = SimpleImputer(strategy = "mean").fit(num_data).transform(num_data)
num_data = StandardScaler(with_mean=False).fit(num_data).transform(num_data)
num_data = pd.DataFrame(num_data)

X = pd.concat([cat_data, num_data], axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state = 0)
y_train = y_train.values.ravel()
y_valid = y_valid.values.ravel()

from itertools import product
from sklearn.metrics import recall_score
from sklearn.svm import SVC

def optimizer(model, parameters):
    best_score = 0
    param_combs = [dict(zip(parameters,values)) for values in product(*parameters.values())] #defining all parameter combinations
    for i in param_combs:
        clf = model(**i).fit(X_train, y_train)  #fitting given model on all parameter combinations
        y_pred = clf.predict(X_valid)
        score = recall_score(y_valid, y_pred)
        print(score)
        if score > best_score:  #choosing best parameters
            best_score = score
            best_parameters = i #remambering the index of the best parameters

    # t0 = time.time()
    # optimized_clf = model(**best_parameters).fit(X_trainval,y_trainval)
    # test_score = optimized_clf.score(X_test,y_test)
    # t1 = time.time()    #testing and timing the optimized model on the testing set
    # run_time = t1 - t0

    # org = y_test
    # pred = optimized_clf.predict(X_test_use)

    print(f'Best parameters: {best_parameters}')
    # print(f'Test set accuracy score with best parameters: {test_score}')
    # print(f"Test set F1 score with best parameters: {f1_score(org, pred, average='weighted')}")
    # print(f'Optimized model run-time: {run_time}')



svc_params = {'kernel': ['rbf'],
              'random_state': [0],
              'C': [0.01, 0.1, 1, 10, 100],
              # 'gamma': [0.01, 0.1, 1, 10, 'auto', 'scale']
              }
model = SVC
print(f'\n------> Results for {model}:')
optimizer(model, svc_params)

clf = SVC(kernel='rbf', random_state=0, C=1, gamma='auto').fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score)
