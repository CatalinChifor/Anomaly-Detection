import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def plot_results(X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, title_prefix):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{title_prefix} Visualization', fontsize=16)

    scenarios = [
        (X_train, y_train, 'Training Data (Ground Truth)', 1),
        (X_test, y_test, 'Test Data (Ground Truth)', 2),
        (X_train, y_train_pred, 'Training Data (Predicted)', 3),
        (X_test, y_test_pred, 'Test Data (Predicted)', 4)
    ]

    for X, y, title, idx in scenarios:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='blue', s=20, alpha=0.6)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='red', marker='^', s=40)
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        if idx == 1:
            ax.legend(['Inliers', 'Outliers'])

    plt.tight_layout()
    plt.show()


contamination = 0.15
n_train = 300
n_test = 200
n_features = 3

X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train,
    n_test=n_test,
    n_features=n_features,
    contamination=contamination,
    random_state=42
)

print("Data:", X_train.shape, X_test.shape)


ocsvm_linear = OCSVM(kernel='linear', contamination=contamination)
ocsvm_linear.fit(X_train)

y_train_pred_lin = ocsvm_linear.predict(X_train)
y_test_pred_lin = ocsvm_linear.predict(X_test)
y_test_scores_lin = ocsvm_linear.decision_function(X_test)

bacc_lin = balanced_accuracy_score(y_test, y_test_pred_lin)
auc_lin = roc_auc_score(y_test, y_test_scores_lin)

print(f"OCSVM Linear - Balanced Accuracy: {bacc_lin:.4f}")
print(f"OCSVM Linear - ROC AUC: {auc_lin:.4f}")

plot_results(
    X_train, y_train,
    X_test, y_test,
    y_train_pred_lin, y_test_pred_lin,
    "OCSVM (Linear)"
)


ocsvm_rbf = OCSVM(kernel='rbf', contamination=contamination)
ocsvm_rbf.fit(X_train)

y_test_pred_rbf = ocsvm_rbf.predict(X_test)
y_test_scores_rbf = ocsvm_rbf.decision_function(X_test)

bacc_rbf = balanced_accuracy_score(y_test, y_test_pred_rbf)
auc_rbf = roc_auc_score(y_test, y_test_scores_rbf)

print(f"OCSVM RBF - Balanced Accuracy: {bacc_rbf:.4f}")
print(f"OCSVM RBF - ROC AUC: {auc_rbf:.4f}")


dsvdd = DeepSVDD(contamination=contamination, epochs=20, verbose=0, random_state=42)

dsvdd.fit(X_train)

y_train_pred_deep = dsvdd.predict(X_train)
y_test_pred_deep = dsvdd.predict(X_test)
y_test_scores_deep = dsvdd.decision_function(X_test)

bacc_deep = balanced_accuracy_score(y_test, y_test_pred_deep)
auc_deep = roc_auc_score(y_test, y_test_scores_deep)

print(f"DeepSVDD - Balanced Accuracy: {bacc_deep:.4f}")
print(f"DeepSVDD - ROC AUC: {auc_deep:.4f}")
plt.show(block=False)
plt.pause(0.1)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
