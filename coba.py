
# === Import Library ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load Dataset ===
df = pd.read_csv("processed_kelulusan.csv")

# Pisahkan fitur (X) dan label (y)
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# === Standarisasi Data ===
sc = StandardScaler()
Xs = sc.fit_transform(X)

# === Split Data ===
X_train, X_temp, y_train, y_temp = train_test_split(Xs, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Shape Data:")
print("Train :", X_train.shape)
print("Valid :", X_val.shape)
print("Test  :", X_test.shape)

# === Bangun Model ===
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Output untuk klasifikasi biner
])

# === Kompilasi Model ===
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", "AUC"]
)

model.summary()

# === Callback Early Stopping ===
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# === Latih Model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# === Evaluasi Model ===
from sklearn.metrics import classification_report, confusion_matrix

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy :", acc)
print("Test AUC      :", auc)

# === Prediksi ===
y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

# === Visualisasi Learning Curve ===
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=120)
plt.show()