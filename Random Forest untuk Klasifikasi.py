import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

# --- Setup Path (Sama kayak P5) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_DIR = os.path.join(BASE_DIR, "model")

print("--- Memulai Pertemuan 6: Deep Dive Random Forest ---")

# --- Langkah 1: Muat Data (Pilihan A) ---
print("\n[Langkah 1: Muat Data]")
df = pd.read_csv(os.path.join(DATASET_DIR, "processed_kelulusan.csv"))
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split ulang 70/15/15 (Pake random_state 42 biar konsisten)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Shape: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# --- Langkah 2: Pipeline & Baseline Random Forest ---
print("\n[Langkah 2: Baseline Random Forest]")
num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# --- Langkah 3: Validasi Silang (BARU) ---
print("\n[Langkah 3: Validasi Silang (K-Fold)]")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (train): {scores.mean():.4f} ± {scores.std():.4f}")

# --- Langkah 4: Tuning Ringkas (GridSearch) ---
print("\n[Langkah 4: Tuning (GridSearch CV)]")
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Parameter terbaik:", gs.best_params_)
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# --- Langkah 5: Evaluasi Akhir (Test Set) ---
print("\n[Langkah 5: Evaluasi Akhir (Test Set)]")
final_model = best_model 
y_test_pred = final_model.predict(X_test)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Plot ROC & PR Curve
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    
    # 1. ROC Curve
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (P6)")
    plot_path_roc = os.path.join(RESULT_DIR, "p6_roc_test.png")
    plt.tight_layout(); plt.savefig(plot_path_roc, dpi=120)
    print(f"Plot ROC disimpan ke {plot_path_roc}")

    # 2. PR Curve (BARU)
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (P6)")
    plot_path_pr = os.path.join(RESULT_DIR, "p6_pr_test.png")
    plt.tight_layout(); plt.savefig(plot_path_pr, dpi=120)
    print(f"Plot PR Curve disimpan ke {plot_path_pr}")
    plt.close('all') # Tutup semua plot

# --- Langkah 6: Pentingnya Fitur (BARU) ---
print("\n[Langkah 6: Feature Importance]")
try:
    # Ambil nama fitur setelah di-preprocess
    feature_names = final_model.named_steps["pre"].get_feature_names_out()
    # Ambil nilai importance dari model RF
    importances = final_model.named_steps["clf"].feature_importances_
    
    top = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("Top feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)
    
# --- Langkah 7: Simpan Model ---
print("\n[Langkah 7: Simpan Model]")
model_path = os.path.join(MODEL_DIR, "rf_model.pkl") # Sesuai nama di Lembar Kerja
joblib.dump(final_model, model_path)
print(f"Model disimpan sebagai {model_path}")

# --- Langkah 8: Cek Inference Lokal ---
print("\n[Langkah 8: Tes Inference Lokal]")
mdl = joblib.load(model_path)
sample = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,      # Fitur dari P4
  "IPK_x_Study": 3.4*7    # Fitur dari P4
}])
print(f"Prediksi sample: {int(mdl.predict(sample)[0])}")

print("\n--- Pertemuan 6 Selesai ---")