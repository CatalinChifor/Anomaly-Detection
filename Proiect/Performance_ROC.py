import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

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

def generate_comparison_table(datasets):
    results = []

    print(f"\n{'Dataset':<20} | {'ECOD':<8} | {'KNN':<8} | {'LOF':<8} | {'IForest':<8}")
    print("-" * 65)

    for ds_name in datasets:
        
            data = np.load(ds_name)
            X, y = data['X'], data['y']
            
            contamination = np.sum(y == 1) / len(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )
            
            classifiers = {
                'ECOD': ECOD(contamination=contamination),
                'KNN': KNN(contamination=contamination),
                'LOF': LOF(contamination=contamination),
                'IForest': IForest(contamination=contamination, random_state=42)
            }
            
            row_result = {'Dataset': ds_name.replace('.npz', '')}
            
            for clf_name, clf in classifiers.items():
                clf.fit(X_train)
                
                y_scores = clf.decision_function(X_test)
                
                roc = roc_auc_score(y_test, y_scores)
                
                row_result[clf_name] = roc

            results.append(row_result)

            print(f"{row_result['Dataset']:<20} | {row_result['ECOD']:.4f}   | {row_result['KNN']:.4f}   | {row_result['LOF']:.4f}   | {row_result['IForest']:.4f}")

        

    if results:
        df = pd.DataFrame(results)
        
        avg_row = df.mean(numeric_only=True)
        
        print("-" * 65)
        print(f"{'AVG':<20} | {avg_row['ECOD']:.4f}   | {avg_row['KNN']:.4f}   | {avg_row['LOF']:.4f}   | {avg_row['IForest']:.4f}")
        
        avg_df = pd.DataFrame([avg_row], index=['AVG'])
        avg_df['Dataset'] = 'AVG' 
        
        df_final = pd.concat([df, avg_df], ignore_index=False)
        
        return df_final
    return None

if __name__ == "__main__":
    df_results = generate_comparison_table(files_list)
    
    if df_results is not None:
        filename = 'tabel_4_roc.csv'
        df_results.to_csv(filename, index=False)
      