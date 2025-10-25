import pandas as pd

# 1. Baca file CSV
df = pd.read_csv("kelulusan_mahasiswa.csv")

# 2. Tampilkan isi d|)
print(df)

# 3. Statistik ringkas
print("\nStatistik Ringkas:")
print(df.describe())

# 4. Simpan dataset ke Excel (sheet 1)
with pd.ExcelWriter("kelulusan_mahasiswa.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Data Mahasiswa", index=False)
    df.describe().to_excel(writer, sheet_name="Statistik")

print("Data & Statistik berhasil disimpan ke Excel: kelulusan_mahasiswa.xlsx")