import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA

data = fetch_openml('shuttle', version=1, as_frame=False)
X, y = data.data, (data.target.astype(int) != 1).astype(int) 

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.6, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

contam = np.mean(y_tr) 

pca = PCA(contamination=contam).fit(X_tr_s)
var_ratio = pca.explained_variance_ratio_

plt.bar(range(len(var_ratio)), var_ratio, label='Individual')
plt.step(range(len(var_ratio)), np.cumsum(var_ratio), where='mid', c='r', label='Cumulative')
plt.legend(); plt.title("Explained Variance"); plt.show()

def evaluate(model, name, xtr, ytr, xte, yte):
    print(f"{name} Bal Acc - Train: {balanced_accuracy_score(ytr, model.labels_):.4f}, "
          f"Test: {balanced_accuracy_score(yte, model.predict(xte)):.4f}")

evaluate(pca, "PCA", X_tr_s, y_tr, X_te_s, y_te)

kpca = KPCA(contamination=contam).fit(X_tr_s)
evaluate(kpca, "KPCA", X_tr_s, y_tr, X_te_s, y_te)