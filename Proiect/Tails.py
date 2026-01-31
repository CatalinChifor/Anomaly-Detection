import numpy as np
import matplotlib.pyplot as plt
from pyod.models.ecod import ECOD
from scipy.stats import scoreatpercentile

def draw_figure():
   
    np.random.seed(42)
    n_inliers = 180
    n_outliers = 20
    contamination = n_outliers / (n_inliers + n_outliers)

    X1 = np.random.normal(loc=5.0, scale=0.5, size=(n_inliers, 2))
    
    X2 = np.random.uniform(low=0, high=3, size=(n_outliers, 2))

    X = np.vstack([X1, X2])
    
    y_true = np.zeros(len(X))
    y_true[-n_outliers:] = 1  

    clf = ECOD(contamination=contamination)
    clf.fit(X)

   
    score_left = -1 * np.sum(np.log(clf.U_l), axis=1)

    score_right = -1 * np.sum(np.log(clf.U_r), axis=1)

    score_avg = (score_left + score_right) / 2

    score_ecod = clf.decision_scores_

    def get_labels(scores, contam):
        threshold = scoreatpercentile(scores, 100 * (1 - contam))
        return (scores > threshold).astype(int)

    y_pred_left = get_labels(score_left, contamination)
    y_pred_right = get_labels(score_right, contamination)
    y_pred_avg = get_labels(score_avg, contamination)
    y_pred_ecod = clf.labels_ 

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    
    titles = [
        "(a) Ground Truth", 
        "(b) Left Tail Only", 
        "(c) Right Tail Only", 
        "(d) Avg of Tails", 
        "(e) ECOD (Skewness Corrected)"
    ]
    
    datasets = [
        (y_true, "Realitate"),
        (y_pred_left, "Left Tail"),
        (y_pred_right, "Right Tail"),
        (y_pred_avg, "Average"),
        (y_pred_ecod, "ECOD Final")
    ]

    for i, ax in enumerate(axes):
        labels, name = datasets[i]
        
        ax.scatter(X[labels == 0, 0], X[labels == 0, 1], c='blue', s=20, label='Normal')
        
        ax.scatter(X[labels == 1, 0], X[labels == 1, 1], c='red', s=20, label='Outlier')
        
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc='upper left')

    plt.suptitle("Importanța Corecției de Asimetrie", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_figure()