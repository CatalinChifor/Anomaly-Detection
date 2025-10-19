import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.metrics import balanced_accuracy_score


contamination_rate = 0.10  
n_samples = 1000          
n_features = 2             

mu = np.array([5, 10])

Sigma = np.array([
    [4, 2],
    [2, 9]
])

x = np.random.normal(loc=0, scale=1, size=(n_samples, n_features))  

L = np.linalg.cholesky(Sigma)

y_normal = x @ L.T + mu

n_outliers = int(n_samples * contamination_rate)
y_true = np.zeros(n_samples, dtype=int)  
y_true[:n_outliers] = 1                 
np.random.shuffle(y_true)               

y_outliers = np.random.normal(loc=30, scale=5, size=(n_outliers, n_features))

y_data = y_normal.copy()
y_data[y_true == 1] = y_outliers

print(f"Generated dataset shape: {y_data.shape}. Contamination: {contamination_rate*100:.0f}%")
print("-" * 50)


sample_mean = np.mean(y_data, axis=0)
sample_cov = np.cov(y_data, rowvar=False)
inv_cov = np.linalg.inv(sample_cov)

mahalanobis_sq_distances = np.zeros(n_samples)
for i in range(n_samples):
    diff = y_data[i] - sample_mean
    mahalanobis_sq_distances[i] = diff.T @ inv_cov @ diff


quantile_level = 1 - contamination_rate  
threshold_mahalanobis_sq = np.quantile(mahalanobis_sq_distances, quantile_level)

print(f"Sample mean (μ̂): {sample_mean}")
print(f"Mahalanobis Squared Distance Threshold (D²): {threshold_mahalanobis_sq:.4f}")
print("-" * 50)


y_pred = (mahalanobis_sq_distances > threshold_mahalanobis_sq).astype(int)

b_accuracy = balanced_accuracy_score(y_true, y_pred)

print(f"Number of predicted anomalies (y_pred=1): {np.sum(y_pred)}")
print(f"Balanced Accuracy of the Mahalanobis method: {b_accuracy:.4f}")
