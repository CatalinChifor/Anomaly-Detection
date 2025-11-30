import numpy as np
import scipy.io
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

file_name = "shuttle.mat"

mat = scipy.io.loadmat(file_name)
X = mat['X']
y = mat['y'].ravel()

y = (y != 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

contamination = np.mean(y_train == 1)

def run_model(model, name):
    model.fit(X_train_scaled)

    y_pred = model.predict(X_test_scaled)
    y_scores = model.decision_function(X_test_scaled)

    ba = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)

    print(f"[{name}] BA: {ba:.4f} | AUC: {auc:.4f}")

run_model(OCSVM(contamination=contamination), "OCSVM")

run_model(DeepSVDD(contamination=contamination, epochs=20, verbose=0), "DeepSVDD Default")

architectures = [
    [32, 16],
    [64, 32],
    [128, 64, 32],
    [100, 50, 25, 10]
]

for arch in architectures:
    model = DeepSVDD(hidden_neurons=arch, contamination=contamination, epochs=20, verbose=0)
    run_model(model, f"DeepSVDD {arch}")
