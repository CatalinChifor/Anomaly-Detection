import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, losses

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

def add_noise(img):
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=1.0)
    return tf.clip_by_value(img + 0.35 * noise, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

class ConvAutoencoder(models.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2),
            layers.Conv2D(4, (3,3), activation='relu', padding='same', strides=2)
        ])
        self.decoder = models.Sequential([
            layers.Conv2DTranspose(4, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(8, 3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))

def show_plot(model, clean, noisy, title):
    if hasattr(clean, 'numpy'): clean = clean.numpy()
    if hasattr(noisy, 'numpy'): noisy = noisy.numpy()

    dec_clean = model.predict(clean[:5])
    dec_noisy = model.predict(noisy[:5])
    
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    rows = [clean[:5], noisy[:5], dec_clean, dec_noisy]
    names = ["Original", "Noisy", "Rec. (Orig)", "Rec. (Noisy)"]
    
    for r_idx, (row_data, name) in enumerate(zip(rows, names)):
        for c_idx in range(5):
            ax = plt.subplot(4, 5, r_idx * 5 + c_idx + 1)
            plt.imshow(row_data[c_idx].squeeze(), cmap='gray')
            if c_idx == 2: ax.set_title(name)
            plt.axis('off')
    plt.show()

print("Training Standard Autoencoder...")
ae = ConvAutoencoder()
ae.compile(optimizer='adam', loss='mse')
ae.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test), verbose=0)

train_loss = tf.reduce_mean(losses.mse(ae.predict(x_train), x_train), axis=[1,2])
threshold = np.mean(train_loss) + np.std(train_loss)

loss_clean = tf.reduce_mean(losses.mse(ae.predict(x_test), x_test), axis=[1,2])
loss_noisy = tf.reduce_mean(losses.mse(ae.predict(x_test_noisy), x_test_noisy), axis=[1,2])
acc = (np.sum(loss_clean < threshold) + np.sum(loss_noisy > threshold)) / (len(x_test)*2)

print(f"Threshold: {threshold:.4f}, Accuracy: {acc*100:.2f}%")
show_plot(ae, x_test, x_test_noisy, "Standard AE Results")

print("Training Denoising Autoencoder...")
denoiser = ConvAutoencoder()
denoiser.compile(optimizer='adam', loss='mse')
denoiser.fit(x_train_noisy, x_train, epochs=10, batch_size=64, validation_data=(x_test_noisy, x_test), verbose=0)
show_plot(denoiser, x_test, x_test_noisy, "Denoising AE Results")