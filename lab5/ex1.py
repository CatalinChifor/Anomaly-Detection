import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.multivariate_normal([5, 10, 2], [[3, 2, 2], [2, 10, 1], [2, 1, 2]], 500)

X_c = X - np.mean(X, axis=0)
vals, vecs = np.linalg.eig(np.cov(X_c, rowvar=False))

idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
X_pca = X_c @ vecs  

fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.scatter(*X.T); ax.set_title("Original Data"); plt.show()

ratio = vals / np.sum(vals)
plt.bar(['PC1','PC2','PC3'], ratio)
plt.step(['PC1','PC2','PC3'], np.cumsum(ratio), where='mid')
plt.title("Explained Variance"); plt.show()

for i in [2, 1]: 
    dev = np.abs(X_pca[:, i] - np.mean(X_pca[:, i]))
    mask = dev > np.quantile(dev, 0.9) 
    
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*X[~mask].T, c='b', label='Normal')
    ax.scatter(*X[mask].T, c='r', label='Anomaly')
    ax.set_title(f"Outliers on PC{i+1}"); plt.show()

norm_dist = np.sqrt(np.sum((X_pca / np.sqrt(vals))**2, axis=1))
mask_dist = norm_dist > np.quantile(norm_dist, 0.9)

fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.scatter(*X[~mask_dist].T, c='g', label='Normal')
ax.scatter(*X[mask_dist].T, c='orange', label='Anomaly')
ax.set_title("Normalized Distance Outliers"); plt.show()