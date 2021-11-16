"""
Welcome to the Datacation Bootcamp!

Today, we present a coding challenge to you.
In this challenge, you will be given two brain tumor dataset. A train and a test dataset.
The target variable has been removed from the test dataset. You will try different Machine Learning models and
use your best model to predict whether patients have a brain tumor or not.

Try to get as far as possible in the following exercises, increasing in difficulty:

1. Loading the training set train_brain.csv and split the training set in a training and validation set, then preprocess the data. Options include:
    - imputing missing values
    - one-hot-encoding categorical values
    - scaling the data
2. Implement the machine learning model called the Support Vector Machine (SVM) and optimize based on the validation accuracy.
3. Visualize the SVM accuracy results in a graph.
4. Use a grid search to find the optimal hyperparameters of the SVM, KNN and RandomForest models using 3-fold cross validation.
5. Visualize the SVM, KNN and RandomForest accuracy results in a heatmap.
6. Implement a Neural Network and visualize the loss and accuracy, both for the test and training dataset.
7. Apply any type of model and preprocessing steps necessary to achieve the highest possible validation accuracy.
8. Use the test_brain.csv dataset to predict whether the patients have a brain tumor or not. The target variable has been removed from the dataset.
   Save the prediction results in a list with the same order as the patients in the test dataset.
   Use the below given code to store the list as .pkl file and save it under the Results section.
   Finally, do a push request to the github repository. We will calculate your final accuracy score.

"""

######################################   EXERCISE 1   ######################################
'''
Loading the training set train_brain.csv and split the training set in a training and validation set, then preprocess the data. Options include:
    - imputing missing values
    - one-hot-encoding categorical values
    - scaling the data
'''

# Read in the dataset: (https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

# Split the training set in a train and validation set, use random_state = 0: (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

# Impute missing values: (https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

# Scale numerical columns: (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)


######################################   EXERCISE 2   ######################################
'''
Implement the machine learning model called the Support Vector Machine (SVM) and optimize based on the validation accuracy.
'''

# Implement the Support Vector Machine: (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)


######################################   EXERCISE 3   ######################################
'''
Visualize the SVM accuracy results in a graph.
'''

# Visualize the SVM results: (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)


######################################   EXERCISE 4   ######################################
'''
Use a grid search to find the optimal hyperparameters of the SVM, KNN and RandomForest models using 3-fold cross validation.
'''

# Make use of GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


######################################   EXERCISE 5   ######################################
'''
Visualize the SVM, KNN and RandomForest accuracy results in a heatmap.
'''

# Make us of a heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html


######################################   EXERCISE 6   ######################################
'''
Implement a basic Neural Network and visualize the loss and accuracy, both for the test and training dataset.
'''

# Make use of the MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier


######################################   EXERCISE 7   ######################################
'''
Apply any type of model and preprocessing steps necessary to achieve the highest possible validation accuracy.
'''


######################################   EXERCISE 8   ######################################
'''
Time to predict using your best ML model!
Use the test_brain.csv dataset to predict whether the patients have a brain tumor or not. The target variable has been removed from the dataset.
Save the prediction results in a list with the same order as the patients in the test dataset.
Use the below given code to store the list as .pkl file and save it under the Results section.
Finally, do a push request to the github repository. We will calculate your final accuracy score.
'''

import pickle

group_number = 0
ypred = []

with open(f'Results/test_predictions_group_{group_number}.pkl', 'wb') as f:
    pickle.dump(ypred, f)