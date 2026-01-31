# Anomaly Detection

This repository contains implementations and experiments for anomaly detection algorithms, with a particular focus on **ECOD (Empirical Cumulative distribution functions for Outlier Detection)**. The code represents a partial replication of the experiments described in the paper:

> **ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions**

## 📁 Repository Structure

### Project Files (`/Proiect`)

The main experimental code for replicating ECOD paper results:

- **`ECOD_var_ROC.py`** - Evaluates different ECOD variants (ECOD-L, ECOD-R, ECOD-B, ECOD) using ROC-AUC metrics across multiple datasets. Replicates Table 2 from the paper.

- **`ECOD_var_AP.py`** - Evaluates ECOD variants using Average Precision (AP) metrics. Replicates Table 3 from the paper.

- **`Performance_ROC.py`** - Compares ECOD against baseline methods (KNN, LOF, IForest) using ROC-AUC scores. Replicates Table 4 from the paper.

- **`Performance_AP.py`** - Compares ECOD against baseline methods using Average Precision metrics. Replicates Table 5 from the paper.

- **`Compare.py`** - Provides a comprehensive comparison framework for ECOD, KNN, and LOF on individual datasets, including visualization of results with accuracy, ROC-AUC, and execution time metrics.

### Lab Files (`/lab2`, `/lab4`, `/lab5`)

Additional experimental code from coursework:

- **`lab2/Ex2.py`** - Visualization exercises
- **`lab4/ex1.py`** - 3D visualization of anomaly detection results on training and test data
- **`lab5/ex4.py`** - Autoencoder-based anomaly detection with noise handling

## 🎯 Datasets

The experiments use 8 standard outlier detection benchmark datasets:

1. `4_breastw.npz` - Breast Cancer Wisconsin
2. `6_cardio.npz` - Cardiotocography
3. `18_Ionosphere.npz` - Ionosphere
4. `23_mammography.npz` - Mammography
5. `29_Pima.npz` - Pima Indians Diabetes
6. `30_satellite.npz` - Satellite
7. `31_satimage-2.npz` - Satimage-2
8. `32_shuttle.npz` - Shuttle

## 🔬 ECOD Variants

The implementation includes several ECOD variants:

- **ECOD-L** (Left-tail): Uses left-tail probabilities
- **ECOD-R** (Right-tail): Uses right-tail probabilities  
- **ECOD-B** (Bilateral): Combines both tails
- **ECOD**: Automatic variant selection (paper's default)

## 🚀 Usage

### Run Performance Comparison

```bash
# Compare ECOD variants with ROC-AUC
python Proiect/ECOD_var_ROC.py

# Compare ECOD variants with Average Precision
python Proiect/ECOD_var_AP.py

# Compare ECOD vs. other methods (ROC-AUC)
python Proiect/Performance_ROC.py

# Compare ECOD vs. other methods (Average Precision)
python Proiect/Performance_AP.py
```

### Compare Models on Single Dataset

```bash
# Run comparison on Shuttle dataset
python Proiect/Compare.py
```

## 📊 Output

Each script generates CSV files containing experimental results:

- `rezultate_tabel_2_extins.csv` - ECOD variants ROC-AUC scores
- `tabel_3_average_precision.csv` - ECOD variants AP scores
- `tabel_4_roc.csv` - Algorithm comparison ROC-AUC scores
- `tabel_5_ap.csv` - Algorithm comparison AP scores

## 🛠️ Dependencies

```
numpy
pandas
scikit-learn
matplotlib
pyod (Python Outlier Detection library)
```

## 📝 Notes

This implementation serves as a partial replication of the ECOD paper experiments. The code uses the PyOD library's ECOD implementation and compares it against established baseline methods (KNN, LOF, IForest) on standard benchmark datasets.

## 📖 Reference

For more details on the ECOD algorithm, please refer to the original paper:
*ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions*