import numpy as np
import scipy.io
import os
import urllib.request
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score

file_name = "cardio.mat"
mat = scipy.io.loadmat(file_name)
X = mat['X']
y = mat['y'].ravel()

y_sklearn = 1 - 2 * y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_sklearn, 
    train_size=0.40, 
    random_state=42, 
    stratify=y_sklearn
)

contamination_rate = np.sum(y_train == -1) / len(y_train)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

param_grid = {
    'ocsvm__kernel': ['rbf', 'linear', 'sigmoid'],
    'ocsvm__gamma': ['scale', 'auto', 0.1],
    'ocsvm__nu': [contamination_rate, 0.05, 0.1, 0.2] 
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Set Balanced Accuracy: {balanced_accuracy_score(y_test, grid_search.predict(X_test)):.4f}")