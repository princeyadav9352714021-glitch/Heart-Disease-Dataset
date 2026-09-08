# Heart Disease Prediction — Case Study

## Overview
This project builds a machine learning model to predict whether a patient has heart disease, based on 13 clinical features. It uses the **Heart Disease Dataset** (`johnsmith88/heart-disease-dataset`) from Kaggle, containing 1,025 patient records, and trains an **XGBoost classifier** to achieve ~83% accuracy.

## Dataset
Source: [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

| Column | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |
| `target` | 1 = heart disease present, 0 = no heart disease (label) |

## Project Workflow
1. **Data Acquisition** — Download the dataset via `kagglehub`.
2. **Exploratory Data Analysis (EDA)** — Check data types, missing values, and visualize every feature's distribution with histograms.
3. **Outlier Detection & Treatment** — Use boxplots and the IQR (interquartile range) method on `trestbps`, `chol`, `thalach`, and `oldpeak`; cap (Winsorize) and filter outliers.
4. **Feature/Target Split** — Separate features (`X`) from the label (`y = target`).
5. **Train/Test Split** — 80% training, 20% testing (`random_state=42` for reproducibility).
6. **Feature Scaling** — Standardize features with `StandardScaler` (fit on train only, applied to both train and test to avoid data leakage).
7. **Model Training** — Train an `XGBClassifier` with tuned hyperparameters:
   - `n_estimators=50`
   - `max_depth=3`
   - `learning_rate=0.05`
   - `subsample=0.8`, `colsample_bytree=0.8`
8. **Evaluation** — Accuracy, precision, recall, and F1-score via `classification_report`.
9. **Model Persistence** — Save the trained model (`model.pkl`) and the fitted scaler (`stscaler.joblib`) with `joblib`.
10. **Inference** — Reload the saved model/scaler and predict on new, unseen patient records.

## Results

| Metric | Class 0 (No Disease) | Class 1 (Disease) |
|---|---|---|
| Precision | 0.89 | 0.79 |
| Recall | 0.75 | 0.90 |
| F1-score | 0.81 | 0.84 |

**Overall Accuracy: ~82.9%**

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
xgboost
joblib
kagglehub
```

Install with:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost joblib kagglehub
```

## How to Run
1. Open `Heart_Disease_Dataset_Explained.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — the dataset downloads automatically via `kagglehub`.
3. The notebook trains the model, evaluates it, and saves `model.pkl` and `stscaler.joblib`.
4. To predict on a new patient, provide their 13 feature values in the same order as the dataset columns (excluding `target`), scale them with the saved scaler, and call `model.predict()`.

## Files
| File | Description |
|---|---|
| `Heart_Disease_Dataset_Explained.ipynb` | Annotated notebook — every code cell has a markdown explanation above it |
| `model.pkl` | Saved trained XGBoost model (generated when the notebook is run) |
| `stscaler.joblib` | Saved `StandardScaler` used to preprocess new inputs (generated when the notebook is run) |

## Notes / Caveats
- This model is trained on a relatively small, single-source dataset (1,025 rows) and is intended for **learning/demonstration purposes only** — it is **not** a medical diagnostic tool and should not be used for real clinical decisions.
- The outlier-filtering step (after capping) is largely redundant since values are already within bounds by that point in the pipeline.
- New input data must always be scaled with the *same* saved scaler (`stscaler.joblib`) before being passed to the model, and in the same feature order as training.
