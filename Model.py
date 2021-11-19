import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle

random.seed(3)
np.random.seed(3)

# Open data
df_train = pd.read_csv('./train_brain.csv')
df_test = pd.read_csv('./test_brain.csv')

# Get most informal variables
variables = ['target']
corr_matrix = df_train.corr()
for j, i in enumerate(corr_matrix['target']):
    if abs(i) > .01 and j != 17:
        variables.append(str(j))

# Slice datasets on variables and compute weights
df_train = df_train[variables]
df_test = df_test[variables[1:]]

# Create separate sets
X_train, y_train = df_train.drop(columns=['target']), df_train[['target']]


def preprocess_data(data_train, data_test):
    numeric_features = data_train.select_dtypes('number').columns
    categorical_features = data_train.select_dtypes(exclude='number').columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    transformer = preprocessor.fit(data_train)
    return transformer.transform(data_train), transformer.transform(data_test)


# Preprocess data
X_train, X_test = preprocess_data(X_train, df_test)

# Create validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3)

# Create model
classifier = XGBClassifier(use_label_encoder=False, verbosity=1, n_jobs=-1, tree_method='hist', eval_metric='logloss',
                           scale_pos_weight=1.8)
# params = {
#     'max_depth': [6, 8, 10],
#     'min_child_weight': [.05, .1, .2],
#     'eta': [.01, .05, .1],
#     'subsample': [.9, .95],
#     'colsample_bytree': [.9, .95],
# }

params = {
    'max_depth': [6],
    'min_child_weight': [.2],
    'eta': [.1],
    'subsample': [.9],
    'colsample_bytree': [.9],
}

# Train with hyperparameter tuning
model = RandomizedSearchCV(estimator=classifier, param_distributions=params, random_state=3)
model.fit(X_train, y_train.to_numpy().flatten())
print(model.best_params_)

# Select and save best model
xgb_model = model.best_estimator_
pickle.dump(xgb_model, open('./Models/Xgb'+'-'+str(time.time()), "wb"))

# Evaluate model on validation data
preds_val = xgb_model.predict(X_val)

print(accuracy_score(y_val, preds_val))
plot_confusion_matrix(xgb_model, X_val, y_val)
plt.show()
plt.clf()

# Predictions on test data
pickle.dump(xgb_model.predict(X_test), open('./Results/Xgb-predictions', "wb"))
