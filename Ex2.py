'''
Name: Andrei Oprea
u796299
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('kanga.csv')
print(df.head())
X = df['X'].to_numpy()
y = df['Y'].to_numpy()
X = np.expand_dims(X, 1)
y = np.expand_dims(y, 1)


#2.2.1
fig, ax = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
ax.scatter(x=X,y= y)
plt.xlabel('Nasal length (mm)')
plt.ylabel('Nasal width (mm)')
plt.title('The relationship between nasal length and nasal width')
# plt.show()

#2.3.1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1, train_size=0.9)

#2.3.2
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print(f'R-squared score: {lr.score(X_test, y_test)}')

#2.3.3
from sklearn.model_selection import cross_val_score
print(f'The mean R-squared score after cross-validation: {np.mean(cross_val_score(lr, X, y, n_jobs=-1))}')

#2.3.4
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

params_svr = {'kernel': ['linear', 'poly', 'rbf'],
              'C': list(range(1,100,10)),
              'epsilon': [0.01, 0.1, 1, 10, 50, 100]}
params_tree = {'max_depth': list(range(1,50))}

svr_rbf = SVR()
tree = DecisionTreeRegressor()

gscv_svr = GridSearchCV(svr_rbf, params_svr, n_jobs=-1).fit(X_train,y_train.ravel())
gscv_tree = GridSearchCV(tree, params_tree, n_jobs=-1).fit(X_train,y_train.ravel())

print(f'R-squared score for GridSearchCV on SVR: {gscv_svr.score(X_test,y_test)}')
print(f'R-squared score for GridSearchCV on Tree Regressor: {gscv_tree.score(X_test,y_test)}')
X_sorted = np.sort(X, axis=0)
ax.plot(X_sorted, gscv_svr.predict(X_sorted), label = f'Support Vector Regression with {gscv_svr.best_params_}')
ax.step(X_sorted, gscv_tree.predict(X_sorted), label = f' Decision Tree Regression with {gscv_tree.best_params_}')
plt.legend()
plt.show()



#2.3.5
#Best performing model based on the R-squared score is the gscv_svr with the score: 0.41530229576254485

#2.4.1
from sklearn import impute
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('2036924_nose.csv')
print(df.head())
df_mean = impute.SimpleImputer(strategy='mean').fit_transform(df)
df_knn = impute.KNNImputer(n_neighbors=3).fit_transform(df)

X_mean = np.expand_dims(df_mean[:,1], 1)
y_mean = np.expand_dims(df_mean[:,2], 1)
X_train_mean, X_test_mean, y_train_mean, y_test_mean = train_test_split(X_mean, y_mean, random_state=0, test_size=0.1, train_size=0.9)
X_knn = np.expand_dims(df_knn[:,1], 1)
y_knn = np.expand_dims(df_knn[:,2], 1)
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, random_state=0, test_size=0.1, train_size=0.9)

lr_mean = LinearRegression().fit(X_test_mean,y_test_mean)
lr_knn = LinearRegression().fit(X_train_knn,y_train_knn)

print(f'R-squared score for mean imputation: {lr_mean.score(X_test_mean,y_test_mean)}')
print(f'R-squared score for knn imputation: {lr_knn.score(X_test_knn,y_test_knn)}')