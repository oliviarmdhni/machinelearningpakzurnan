# kelulusan_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = {
    "IPK": [3.8, 2.5, 3.4, 2.1, 3.9, 2.8, 3.2, 2.7, 3.6, 2.3],
    "Jumlah_Absensi": [3, 8, 4, 12, 2, 6, 5, 7, 4, 9],
    "Waktu_Belajar_Jam": [10, 5, 7, 2, 12, 4, 8, 3, 9, 4],
    "Lulus": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df.to_csv("kelulusan_mahasiswa.csv", index=False)
print("✅ Dataset 'kelulusan_mahasiswa.csv' berhasil dibuat.")


df = pd.read_csv("kelulusan_mahasiswa.csv")
print("\n[INFO DATASET]")
print(df.info())
print(df.head())


print("\n[MISSING VALUE]")
print(df.isnull().sum())

df = df.drop_duplicates()


sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK")
plt.show()


print("\n[STATISTIK DESKRIPTIF]")
print(df.describe())


sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.xlabel("IPK")
plt.ylabel("Frekuensi")
plt.show()


sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar")
plt.xlabel("IPK")
plt.ylabel("Jam Belajar per Minggu")
plt.show()


sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()


df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']


df.to_csv("processed_kelulusan.csv", index=False)
print("✅ Dataset baru 'processed_kelulusan.csv' berhasil disimpan.")


X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Train (70%) dan Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Validation (15%) dan Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n[UKURAN DATASET]")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)