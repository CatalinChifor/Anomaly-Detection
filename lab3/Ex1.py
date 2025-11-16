import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X_train, _ = make_blobs(n_samples=500, n_features=2, centers=1, cluster_std=1.0)
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)


n_projections = 5
n_bins = 30        
range_padding = 2  


projections = np.random.multivariate_normal(
    mean=[0, 0],
    cov=np.eye(2),
    size=n_projections
)

projections /= np.linalg.norm(projections, axis=1, keepdims=True)

hist_bins = []
hist_probs = []

for w in projections:
    proj_vals = X_train @ w
    
    rmin, rmax = proj_vals.min(), proj_vals.max()
    hrange = (rmin - range_padding, rmax + range_padding)

    counts, bin_edges = np.histogram(proj_vals, bins=n_bins, range=hrange)
    
    probs = counts / counts.sum()
    
    hist_bins.append(bin_edges)
    hist_probs.append(probs)


X_test = np.random.uniform(-3, 3, size=(500, 2))

scores = np.zeros(len(X_test))

for i, x in enumerate(X_test):
    proj_scores = []
    for w, bins, probs in zip(projections, hist_bins, hist_probs):
        val = x @ w
        idx = np.searchsorted(bins, val) - 1
        
        if idx < 0 or idx >= len(probs):
            p = 1e-6
        else:
            p = probs[idx] if probs[idx] > 0 else 1e-6
        
        proj_scores.append(p)
    
    scores[i] = np.mean(proj_scores)

anomaly_intensity = -np.log(scores + 1e-12)


plt.figure(figsize=(6, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_intensity, cmap='viridis')
plt.colorbar(label="Anomaly Score")
plt.title("Simplified LODA — Score Map")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
