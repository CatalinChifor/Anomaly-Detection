import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF



data_set = 'Shuttle'
def compare_models(file_name='4_breastw.npz'):
    print(f"--- Încărcare date: {file_name} ---")
    
    data = np.load(file_name)
    X, y = data['X'], data['y']
        
    contamination = np.sum(y == 1) / len(y)
    print(f"Contaminare: {contamination:.2%}")
    print(f"Dimensiuni: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifiers = {
        'ECOD': ECOD(contamination=contamination),
        'KNN': KNN(contamination=contamination, n_neighbors=5), 
        'LOF': LOF(contamination=contamination)
    }

    results = {'Model': [], 'Accuracy': [], 'ROC-AUC': [], 'Time (s)': []}

    print("\n--- Start Comparare ---")
    
    for model_name, clf in classifiers.items():
        print(f"Rulez {model_name}...")
        start_time = time.time()
        
        clf.fit(X_train)
        
        y_pred = clf.predict(X_test)            
        y_scores = clf.decision_function(X_test) 
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_scores)
        duration = time.time() - start_time
        
        results['Model'].append(model_name)
        results['Accuracy'].append(acc)
        results['ROC-AUC'].append(roc)
        results['Time (s)'].append(duration)

    print("\n" + "="*50)
    print(f"{'Model':<10} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Time (s)':<10}")
    print("-" * 50)
    for i in range(len(results['Model'])):
        print(f"{results['Model'][i]:<10} | "
              f"{results['Accuracy'][i]:.4f}     | "
              f"{results['ROC-AUC'][i]:.4f}    | "
              f"{results['Time (s)'][i]:.4f}")
    print("="*50)

    vizualizare_rezultate(results)

def vizualizare_rezultate(results):
    x = np.arange(len(results['Model']))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, results['Accuracy'], width, label='Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, results['ROC-AUC'], width, label='ROC-AUC', color='orange')

    ax.set_ylabel('Scor')
    ax.set_title(f'Comparare Performanță pentru data-set {data_set}: ECOD vs KNN vs LOF ')
    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'])
    ax.set_ylim(0, 1.1) 
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_models('32_shuttle.npz')
   