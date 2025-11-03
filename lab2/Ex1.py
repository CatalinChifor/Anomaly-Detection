import numpy as np
import matplotlib.pyplot as plt

weight= np.random.randn(1)
intercept=np.random.randn()
samples=100

m=1
s=4

epsilon=np.random.normal(loc=m,scale=np.sqrt(s),size=samples)

x=np.random.randn(samples,1)
linear=x @ weight + intercept
y=linear+epsilon
print(f"Target Vector (Y) shape: {y.shape}")
plt.figure(figsize=(10, 7))

plt.scatter(x.flatten(), y, label='Generated Data Points (with Noise)', color='blue', alpha=0.6)

x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100).reshape(-1, 1)
y_line_perfect = x_line @ weight + intercept

X = np.hstack([x, np.ones_like(x)])
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage_scores = np.diag(H)


high_lev_idx = np.argmax(leverage_scores)

plt.figure(figsize=(10, 7))
plt.scatter(x.flatten(), y, label='Data Points', color='blue', alpha=0.6)
plt.scatter(x[high_lev_idx], y[high_lev_idx], color='red', s=120, marker='*', label='Highest leverage point')

x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100).reshape(-1, 1)
y_line_perfect = x_line @ weight + intercept
plt.plot(x_line.flatten(), y_line_perfect, color='black', linewidth=2, label='True Linear Model')

plt.title('1D Linear Model with Gaussian Noise & Leverage Scores')
plt.xlabel('Feature X')
plt.ylabel('Target Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Highest leverage score: {leverage_scores[high_lev_idx]:.3f} (point index {high_lev_idx})")

#EX1 2D

def TwoDim_Ex(samples=100, mu=0, sigma2=1.0, seed=42):
    
    np.random.seed(seed)

    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()

    x1 = np.random.randn(samples, 1)
    x2 = np.random.randn(samples, 1)

    epsilon = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=samples)

    y = a * x1.flatten() + b * x2.flatten() + c + epsilon

    X = np.hstack([x1, x2, np.ones_like(x1)]) 
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage_scores = np.diag(H)

    high_lev_idx = np.argmax(leverage_scores)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x1, x2, y, c=leverage_scores, cmap='viridis', s=60, alpha=0.8)
    ax.scatter(x1[high_lev_idx], x2[high_lev_idx], y[high_lev_idx],
               color='red', s=150, marker='*', label='Highest leverage point')

    ax.set_title("2D Linear Model with Gaussian Noise & Leverage Scores")
    ax.set_xlabel("Feature X1")
    ax.set_ylabel("Feature X2")
    ax.set_zlabel("Target Y")
    ax.legend()

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Leverage Score")

    plt.show()

    print(f"Highest leverage score: {leverage_scores[high_lev_idx]:.3f} (point index {high_lev_idx})")
    print(f"Leverage score range: {leverage_scores.min():.3f} – {leverage_scores.max():.3f}")
    print(f"Model parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}")