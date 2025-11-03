import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    contamination=0.1,
    random_state=42
)

neighbor_values = [1, 5, 10, 20]

for n_neighbors in neighbor_values:
    print(f"\n=== KNN with n_neighbors = {n_neighbors} ===")

    clf = KNN(n_neighbors=n_neighbors)
    clf.fit(X_train)

    y_train_pred = clf.labels_              
    y_test_pred = clf.predict(X_test)       

    train_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_acc = balanced_accuracy_score(y_test, y_test_pred)
    print(f"Balanced accuracy (train): {train_acc:.3f}")
    print(f"Balanced accuracy (test):  {test_acc:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [
        "Ground Truth (Train)",
        "Predicted (Train)",
        "Ground Truth (Test)",
        "Predicted (Test)"
    ]
    datasets = [
        (X_train, y_train),
        (X_train, y_train_pred),
        (X_test, y_test),
        (X_test, y_test_pred)
    ]

    for ax, (X, y), title in zip(axes.ravel(), datasets, titles):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Inliers", alpha=0.6)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Outliers", alpha=0.7)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"KNN Anomaly Detection (n_neighbors = {n_neighbors})", fontsize=14)
    plt.tight_layout()
    plt.show()
