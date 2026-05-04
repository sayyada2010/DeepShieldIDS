# 🛡️ DeepShieldIDS

### An AI-Powered Intrusion Detection System Using HybridIDSNet (CNN + LSTM + Attention)

---

## 📌 Overview

DeepShieldIDS is a deep learning-based intrusion detection framework designed to detect anomalous network traffic using a hybrid architecture called **HybridIDSNet**, which integrates:

* **Convolutional Neural Networks (CNN)** → Spatial feature extraction
* **LSTM/GRU** → Temporal dependency learning
* **Attention Mechanism** → Feature importance weighting

The system is evaluated on benchmark datasets and supports cross-dataset generalization, making it robust for real-world deployment scenarios.

---

## 🚀 Key Features

* Hybrid deep learning model (CNN + LSTM + Attention)
* Cross-dataset validation (generalization capability)
* SMOTE-based class imbalance handling
* Feature harmonization across datasets
* Baseline comparison (RF, SVM, XGBoost, DL models)
* Explainability support (Attention + SHAP-ready)
* Real-time deployment via FastAPI
* Modular and extensible architecture

---

## 📂 Project Structure

```
DeepShieldIDS/
│
├── data/                     # Raw and processed datasets
├── src/                      # Source code modules
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_harmonization.py
│   ├── smote_balancing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_dataset_eval.py
│   ├── baselines.py
│   ├── explainability.py
│   └── api.py
│
├── outputs/                  # Models, metrics, figures
├── main.py                   # End-to-end pipeline
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

This project uses two standard intrusion detection datasets:

* CIC-IDS2017
* UNSW-NB15

### 📥 Dataset Download Links

(Download manually)

* CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place datasets inside:

```
data/raw/
```

---

## ⚙️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/DeepShieldIDS.git
cd DeepShieldIDS
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Full Pipeline

```bash
python main.py
```

### Pipeline Includes:

1. Data loading
2. Preprocessing
3. Feature harmonization
4. SMOTE balancing
5. Model training
6. Evaluation
7. Cross-dataset validation

---

## 🧠 Model Architecture (HybridIDSNet)

```
Input Features
   ↓
Conv1D (Spatial Features)
   ↓
BatchNorm + ReLU + Pooling
   ↓
BiLSTM / GRU (Temporal Patterns)
   ↓
Attention Layer (Feature Weighting)
   ↓
Dense Layers
   ↓
Output (Attack / Normal)
```

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* False Positive Rate (FPR)

---

## 🔁 Cross-Dataset Validation

| Training Dataset | Testing Dataset |
| ---------------- | --------------- |
| CIC-IDS2017      | CIC-IDS2017     |
| UNSW-NB15        | UNSW-NB15       |
| CIC-IDS2017      | UNSW-NB15       |
| UNSW-NB15        | CIC-IDS2017     |

---

## 🆚 Baseline Models

The framework compares HybridIDSNet against:

* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* CNN-only
* LSTM-only
* CNN-LSTM (without Attention)

---

## 🔍 Explainability

* Attention weight visualization
* Feature importance analysis
* SHAP integration (optional)

---

## 🌐 API Deployment

### Run FastAPI Server

```bash
uvicorn src.api:app --reload
```

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "features": [0.1, 0.5, 0.3]
}
```

### Sample Response

```json
{
  "prediction": "attack",
  "confidence": 0.96
}
```

---

## 📊 Outputs

Generated in:

```
outputs/
```

Includes:

* Trained models
* Evaluation metrics
* Confusion matrices
* ROC curves
* Performance comparison tables

---

## 🧪 Experimental Setup

* Python 3.9
* TensorFlow / PyTorch
* scikit-learn
* imbalanced-learn
* GPU recommended

---

## ⚠️ Important Notes

* SMOTE is applied only on training data
* Cross-dataset validation ensures generalization
* Feature harmonization avoids dataset bias

---

## 📌 Future Work

* Adversarial attack robustness
* Online learning for adaptive IDS
* Sequence-aware data augmentation
* Edge deployment optimization
* Explainable AI enhancements

---

## 📜 License

This project is intended for academic and research purposes.
(Add license if needed: MIT / Apache 2.0)

---

## 🤝 Contribution

Contributions are welcome!

```
fork → modify → pull request
```

---



---

## ⭐ Citation

If you use this work, please cite:

**DeepShieldIDS: An AI-Powered Intrusion Detection System Leveraging HybridIDSNet for Robust Network Security**
