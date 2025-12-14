import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



data = scipy.io.loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

y = (y != 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



class Autoencoder(keras.Model):
    def __init__(self, d):
        super().__init__()
        self.encoder = keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(5, activation="relu"),
            layers.Dense(3, activation="relu"),
        ])
        self.decoder = keras.Sequential([
            layers.Dense(5, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(d, activation="sigmoid"),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))


model = Autoencoder(X_train.shape[1])

model.compile(optimizer="adam", loss="mse")

history = model.fit(
    X_train, X_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, X_test),
    verbose=0
)


plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

train_err = np.mean((X_train - model.predict(X_train))**2, axis=1)
test_err  = np.mean((X_test  - model.predict(X_test))**2, axis=1)

contamination = np.mean(y_train)
threshold = np.quantile(train_err, 1 - contamination)

y_train_pred = (train_err > threshold).astype(int)
y_test_pred  = (test_err  > threshold).astype(int)

print("Train balanced accuracy:",
      balanced_accuracy_score(y_train, y_train_pred))
print("Test balanced accuracy:",
      balanced_accuracy_score(y_test, y_test_pred))
