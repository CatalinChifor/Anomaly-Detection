import warnings
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

def load_odds_dataset(file_path="shuttle.mat"):
   
    try:
        data_mat = loadmat(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    features = data_mat.get("X")
    labels = data_mat.get("y")
    
    if features is None or labels is None:
        raise KeyError(f"Expected keys 'X' and 'y' in {file_path}. Found keys: {list(data_mat.keys())}")
        
    labels = labels.ravel().astype(int)
    return features.astype(float), labels

X_features, y_labels = load_odds_dataset("shuttle.mat")

def train_anomaly_detectors(X_train_set, y_train_set, model_seed=0, dif_config=(64, 32), loda_param=30, enable_dif=True):
   
    
    anomaly_ratio = float(np.mean(y_train_set))

    iforest_model = IForest(contamination=anomaly_ratio, random_state=model_seed).fit(X_train_set)

    if enable_dif:
       
          
                dif_model = DIF(contamination=anomaly_ratio, 
                                  random_state=model_seed,
                                  n_ensemble=2,
                                  n_estimators=10).fit(X_train_set)
            
    else:
        dif_model = None

    loda_model = LODA(contamination=anomaly_ratio, n_bins=loda_param).fit(X_train_set)

    fitted_models = {"IForest": iforest_model, "LODA": loda_model}
    if dif_model is not None:
        fitted_models["DIF"] = dif_model
    
    return fitted_models

def evaluate_models(fitted_models, X_test_set, y_test_set):
    
    evaluation_metrics = {}
    for model_name, model_object in fitted_models.items():
        binary_predictions = model_object.predict(X_test_set)
        
        anomaly_scores = model_object.decision_function(X_test_set)
        
        ba_score = balanced_accuracy_score(y_test_set, binary_predictions)
        
        try:
            auc_score = roc_auc_score(y_test_set, anomaly_scores)
        except ValueError:
            auc_score = np.nan 
            
        evaluation_metrics[model_name] = (ba_score, auc_score)
    return evaluation_metrics


EXPERIMENT_SEEDS = range(10)
final_results = { "IForest": {"BA": [], "AUC": []},
                  "DIF":     {"BA": [], "AUC": []},
                  "LODA":    {"BA": [], "AUC": []} }

print("Starting evaluation with 10 different train-test splits...")

for iteration, random_seed in enumerate(EXPERIMENT_SEEDS):
    print(f"\nSplit {iteration+1}/10 (seed={random_seed})...")
    
    X_train_raw, X_test_raw, y_train_labels, y_test_labels = train_test_split(
        X_features, y_labels, test_size=0.40, random_state=random_seed, stratify=y_labels
    )

    data_scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = data_scaler.transform(X_train_raw)
    X_test_scaled = data_scaler.transform(X_test_raw)

    fitted_detectors = train_anomaly_detectors(
        X_train_scaled, 
        y_train_labels, 
        model_seed=random_seed, 
        dif_config=(32, 16), 
        loda_param=30, 
        enable_dif=True
    )
    
    split_metrics = evaluate_models(fitted_detectors, X_test_scaled, y_test_labels)

    for model_name, (ba_val, auc_val) in split_metrics.items():
        final_results[model_name]["BA"].append(ba_val)
        final_results[model_name]["AUC"].append(auc_val)
    
    print(f"  Split {iteration+1} completed.")

def calculate_mean_std(data_array):
    
    data_array = np.asarray(data_array, dtype=float)
    return np.nanmean(data_array), np.nanstd(data_array)

print("\n=== Shuttle Dataset Evaluation (10 Splits, 40% Test) ===")
for model_key in ["IForest", "DIF", "LODA"]:
    if len(final_results[model_key]["BA"]) > 0:
        ba_avg, ba_std_dev = calculate_mean_std(final_results[model_key]["BA"])
        auc_avg, auc_std_dev = calculate_mean_std(final_results[model_key]["AUC"])
        print(f"{model_key:8s}  BA:    {ba_avg:.3f} \u00B1 {ba_std_dev:.3f}  | ROC-AUC: {auc_avg:.3f} \u00B1 {auc_std_dev:.3f}")
    else:
        print(f"{model_key:8s}  No results available (model was skipped)")