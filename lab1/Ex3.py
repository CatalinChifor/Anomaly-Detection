import numpy as np
from sklearn.metrics import balanced_accuracy_score

def generate_data(n_samples=1000, contamination=0.1):
    n_anomalies = int(n_samples * contamination)
    n_normals = n_samples - n_anomalies

    normal_data = np.random.normal(loc=0, scale=1, size=n_normals)

    anomaly_data = np.random.uniform(low=3, high=5, size=n_anomalies)
    X = np.concatenate([normal_data, anomaly_data])
    y = np.array([0] * n_normals + [1] * n_anomalies)  

    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def detect_anomalies_zscore(X, contamination):
    mean = np.mean(X)
    std = np.std(X)
    z_scores = np.abs((X - mean) / std)

   
    threshold = np.quantile(z_scores, 1 - contamination)

   
    y_pred = (z_scores > threshold).astype(int)

    return y_pred


X_train, y_true = generate_data(n_samples=1000, contamination=0.1)
y_pred = detect_anomalies_zscore(X_train, contamination=0.1)

balanced_acc = balanced_accuracy_score(y_true, y_pred)

print(f"Balanced Accuracy: {balanced_acc:.4f}")
