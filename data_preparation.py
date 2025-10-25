# Langkah 2 — Collection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("kelulusan_mahasiswa.csv")
print("=== INFO DATA ===")
print(df.info())
print(df.head())

# Langkah 3 — Cleaning
print("\n=== CEK MISSING VALUES ===")
print(df.isnull().sum())

df = df.drop_duplicates()

sns.boxplot(x=df['IPK'])
plt.title("Boxplot IPK (Deteksi Outlier)")
plt.show()

# Langkah 4 — EDA
print("\n=== Statistik Deskriptif ===")
print(df.describe())

sns.histplot(df['IPK'], bins=10, kde=True)
plt.title("Distribusi IPK")
plt.show()

sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar (Lulus/Tidak)")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Langkah 5 — Feature Engineering
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']

df.to_csv("processed_kelulusan.csv", index=False)
print("\nDataset baru disimpan sebagai processed_kelulusan.csv")

# Langkah 6 — Splitting
X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("\n=== Split Dataset ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)
