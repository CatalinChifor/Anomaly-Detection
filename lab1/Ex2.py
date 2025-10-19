
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, RocCurveDisplay

contamination = 0.1  
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                           n_redundant=10, n_clusters_per_class=1,
                           weights=[1 - contamination], flip_y=0,
                           random_state=42)

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = KNN(contamination=contamination)
clf.fit(X_train)

y_train_pred = clf.predict(X_train)  
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test) 
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title("ROC Curve - KNN (contamination = 0.1)")
plt.grid()
plt.show()
clf_new = KNN(contamination=0.2)
clf_new.fit(X_train)

y_test_pred_new = clf_new.predict(X_test)
y_test_scores_new = clf_new.decision_function(X_test)

tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_test_pred_new).ravel()
balanced_acc2 = balanced_accuracy_score(y_test, y_test_pred_new)

print("\n--- With Contamination = 0.2 ---")
print(f"True Negatives: {tn2}")
print(f"False Positives: {fp2}")
print(f"False Negatives: {fn2}")
print(f"True Positives: {tp2}")
print(f"Balanced Accuracy: {balanced_acc2:.4f}")

fpr2, tpr2, _ = roc_curve(y_test, y_test_scores_new)
RocCurveDisplay(fpr=fpr2, tpr=tpr2).plot()
plt.title("ROC Curve - KNN (contamination = 0.2)")
plt.grid()
plt.show()
