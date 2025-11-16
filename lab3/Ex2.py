import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA


centers = [(10, 0), (0, 10)]
X_train, _ = make_blobs(
    n_samples=[500, 500],
    centers=centers,
    cluster_std=1.0,
    n_features=2
)

X_test = np.random.uniform(-10, 20, size=(1000, 2))


def plot_scores(X, scores, title, ax):
    im = ax.scatter(X[:, 0], X[:, 1], c=scores, cmap="viridis", s=20)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.colorbar(im, ax=ax)


iforest = IForest(contamination=0.02)
iforest.fit(X_train)

scores_if = iforest.decision_function(X_test)  
anomaly_if = -scores_if                        


dif = DIF(contamination=0.02, hidden_neurons=[32])
dif.fit(X_train)
scores_dif = dif.decision_function(X_test)
anomaly_dif = -scores_dif

loda = LODA(contamination=0.02, n_bins=30, n_random_cuts=100)
loda.fit(X_train)
scores_loda = loda.decision_function(X_test)
anomaly_loda = -scores_loda


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_scores(X_test, anomaly_if, "Isolation Forest", axes[0])
plot_scores(X_test, anomaly_dif, "Deep Isolation Forest (DIF)", axes[1])
plot_scores(X_test, anomaly_loda, "LODA", axes[2])

plt.tight_layout()
plt.show()
