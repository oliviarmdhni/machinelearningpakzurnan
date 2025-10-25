import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import joblib
from flask import Flask, request, jsonify

# --- Setup Path (Biar rapi) ---
# Ini adalah path ke folder P4 (induknya folder 'src' ini)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Definisikan path ke folder lain
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Pastikan folder output ada (buat jaga-jaga)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Memulai Pertemuan 5: Modeling ---")

# --- Langkah 1: Muat Data (Pilihan B dari Lembar Kerja) ---
print("\n[Langkah 1: Muat Data]")
df = pd.read_csv(os.path.join(DATASET_DIR, "processed_kelulusan.csv"))
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split ulang 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Shape: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# --- Langkah 2: Baseline Model & Pipeline ---
print("\n[Langkah 2: Baseline (Logistic Regression)]")
# Ambil semua kolom numerik
num_cols = X_train.select_dtypes(include="number").columns

# Bikin pipeline preprocessing
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), # Isi data kosong
                      ("sc", StandardScaler())]), num_cols),   # Scaling data
], remainder="drop") # Abaikan kolom non-numerik (kalau ada)

# Model Baseline: Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)]) # Gabung preprocessor + model

# Latih baseline
pipe_lr.fit(X_train, y_train)
y_val_pred = pipe_lr.predict(X_val) # Tes di data validasi
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# --- Langkah 3: Model Alternatif (Random Forest) ---
print("\n[Langkah 3: Alternatif (Random Forest)]")
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)]) # Pipeline yg sama, model beda

# Latih RF
pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))

# --- Langkah 4: Validasi Silang & Tuning Ringkas ---
print("\n[Langkah 4: Tuning (GridSearch CV on RF)]")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Parameter yg mau di-tuning
param = {
  "clf__max_depth": [None, 12, 20],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train) # Mulai tuning...

print("Parameter terbaik:", gs.best_params_)
print("Skor F1 CV terbaik:", gs.best_score_)

best_rf = gs.best_estimator_ # Ambil model terbaik hasil tuning
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# --- Langkah 5: Evaluasi Akhir (Test Set) ---
print("\n[Langkah 5: Evaluasi Akhir (Test Set)]")
final_model = best_rf  # Kita putuskan model RF terbaik yg jadi finalis
y_test_pred = final_model.predict(X_test) # UJI DI DATA TEST (sekali aja!)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Bikin plot ROC-AUC
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    try:
        print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
    except:
        pass
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
    
    # Simpan plot ke folder 'result'
    plot_path = os.path.join(RESULT_DIR, "roc_test.png")
    plt.tight_layout(); plt.savefig(plot_path, dpi=120)
    print(f"Plot ROC disimpan ke {plot_path}")

# --- Langkah 6 (Opsional): Simpan Model ---
print("\n[Langkah 6: Simpan Model]")
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(final_model, model_path)
print(f"Model tersimpan ke {model_path}")

# --- Langkah 7 (Opsional): Endpoint Inference (Flask) ---
print("\n[Langkah 7: Menjalankan API Flask]")
app = Flask(__name__)
MODEL = joblib.load(model_path) # Load model yg baru disimpan

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)  # Terima data JSON
    X = pd.DataFrame([data])             # Ubah jadi DataFrame
    yhat = MODEL.predict(X)[0]           # Prediksi
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(X)[:,1][0])
    
    # Kirim balik hasil prediksi
    return jsonify({"prediction": int(yhat), "proba": proba})

if __name__ == "__main__":
    print("Server Flask berjalan di http://127.0.0.1:5000/")
    print("Gunakan aplikasi seperti Postman untuk tes POST ke /predict")
    print("Tekan CTRL+C untuk berhenti.")
    app.run(port=5000, debug=False)