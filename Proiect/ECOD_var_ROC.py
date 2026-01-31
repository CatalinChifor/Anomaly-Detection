import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pyod.models.ecod import ECOD

files_list = [
    '4_breastw.npz',
    '6_cardio.npz',
    '18_Ionosphere.npz',     
    '23_mammography.npz',
    '29_Pima.npz',           
    '30_satellite.npz',
    '31_satimage-2.npz',
    '32_shuttle.npz'
]

def calculate_variant_scores_numpy(X_train, X_test):
   
    n_test, d = X_test.shape
    n_train = X_train.shape[0]
    
    U_l = np.zeros((n_test, d))
    U_r = np.zeros((n_test, d))
    
    for i in range(d):
        sorted_train = np.sort(X_train[:, i])
        
        indices = np.searchsorted(sorted_train, X_test[:, i], side='right')
        
        prob_l = indices / n_train
        
        prob_r = 1.0 - prob_l
        
        prob_l = np.clip(prob_l, 1e-10, 1 - 1e-10)
        prob_r = np.clip(prob_r, 1e-10, 1 - 1e-10)
        
        U_l[:, i] = prob_l
        U_r[:, i] = prob_r

    score_L = np.sum(-np.log(U_l), axis=1)
    score_R = np.sum(-np.log(U_r), axis=1)
    score_B = (score_L + score_R) / 2
    
    return score_L, score_R, score_B

def generate_full_table(datasets):
    results = []

    print(f"\n{'Dataset':<20} | {'ECOD-L':<8} | {'ECOD-R':<8} | {'ECOD-B':<8} | {'ECOD':<8}")
    print("-" * 65)

    for ds_name in datasets:
        try:
            data = np.load(ds_name)
            X, y = data['X'], data['y']
            
            contamination = np.sum(y == 1) / len(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )
            
            clf = ECOD(contamination=contamination)
            clf.fit(X_train)
            score_Auto = clf.decision_function(X_test)
            
            score_L, score_R, score_B = calculate_variant_scores_numpy(X_train, X_test)
            
            roc_l = roc_auc_score(y_test, score_L)
            roc_r = roc_auc_score(y_test, score_R)
            roc_b = roc_auc_score(y_test, score_B)
            roc_auto = roc_auc_score(y_test, score_Auto)
            
            results.append({
                'Data': ds_name.replace('.npz', ''),
                'ECOD-L': roc_l,
                'ECOD-R': roc_r,
                'ECOD-B': roc_b,
                'ECOD': roc_auto
            })

            print(f"{ds_name.replace('.npz', ''):<20} | {roc_l:.4f}   | {roc_r:.4f}   | {roc_b:.4f}   | {roc_auto:.4f}")

        except FileNotFoundError:
            print(f"Eroare: {ds_name} nu este în folder.")
        except Exception as e:
            print(f"Eroare la {ds_name}: {e}")

    if results:
        df = pd.DataFrame(results)
        avg_row = df.mean(numeric_only=True)
        print("-" * 65)
        print(f"{'AVG':<20} | {avg_row['ECOD-L']:.4f}   | {avg_row['ECOD-R']:.4f}   | {avg_row['ECOD-B']:.4f}   | {avg_row['ECOD']:.4f}")
        
        return df
    return None

if __name__ == "__main__":
    df_results = generate_full_table(files_list)
    
    if df_results is not None:
        df_results.to_csv('rezultate_tabel_2_extins.csv', index=False)
       