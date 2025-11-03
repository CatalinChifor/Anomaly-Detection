import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

def knn_vs_lof_demo(n_neighbors=10, contamination=0.07, seed=42):

    np.random.seed(seed)

    X1, _ = make_blobs(
        n_samples=200, centers=[(-10, -10)], cluster_std=2, random_state=seed
    )
    X2, _ = make_blobs(
        n_samples=100, centers=[(10, 10)], cluster_std=6, random_state=seed + 1
    )

    X = np.vstack((X1, X2))

    knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
    lof = LOF(n_neighbors=n_neighbors, contamination=contamination)

    knn.fit(X)
    lof.fit(X)

    y_pred_knn = knn.labels_
    y_pred_lof = lof.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(X[y_pred_knn == 0, 0], X[y_pred_knn == 0, 1],
                    c="blue", s=50, label="Inliers", alpha=0.6)
    axes[0].scatter(X[y_pred_knn == 1, 0], X[y_pred_knn == 1, 1],
                    c="red", s=60, label="Outliers", alpha=0.8)
    axes[0].set_title(f"KNN (n_neighbors={n_neighbors})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(X[y_pred_lof == 0, 0], X[y_pred_lof == 0, 1],
                    c="blue", s=50, label="Inliers", alpha=0.6)
    axes[1].scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1],
                    c="red", s=60, label="Outliers", alpha=0.8)
    axes[1].set_title(f"LOF (n_neighbors={n_neighbors})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Comparison of KNN vs LOF for Different Cluster Densities", fontsize=14)
    plt.tight_layout()
    plt.show()

for k in [5, 10, 20]:
    knn_vs_lof_demo(n_neighbors=k)
