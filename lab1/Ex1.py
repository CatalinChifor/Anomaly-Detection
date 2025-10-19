from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np
X_train, X_test, y_train, y_test = generate_data(
    n_train=500, 
    n_test=300, 
    n_features=2, 
    contamination=0.1
) 
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=30, alpha=0.8)

plt.show()