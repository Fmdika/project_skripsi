# Prediksi IHSG dengan Multivariate Dual-Branch LSTM

Repositori ini berisi artefak penelitian skripsi (S1) dengan judul:

> **Analisis Prediksi Indeks Harga Saham Gabungan (IHSG) Menggunakan Multivariate Long Short-Term Memory (LSTM) Berdasarkan Data Historis dan Variabel Makroekonomi.**

Penelitian menganalisis prediksi IHSG **satu langkah ke depan (one-step ahead)** menggunakan model **Multivariate LSTM** berarsitektur **dual-branch**, lalu membandingkannya secara **objektif** terhadap **baseline Univariate LSTM** pada periode pengujian yang sama. Penelitian tidak mengasumsikan salah satu pendekatan pasti lebih unggul, melainkan mengujinya secara empiris menggunakan metrik dan uji yang terdefinisi.

---

## Ringkasan Penelitian

| Aspek | Keterangan |
|---|---|
| **Objek** | Indeks Harga Saham Gabungan (IHSG), harian |
| **Periode** | 2015-01-02 s/d 2024-12-30 |
| **Target** | Nilai IHSG satu langkah ke depan (t+1) |
| **Model utama** | Multivariate Dual-Branch LSTM |
| **Baseline** | Univariate LSTM (hanya IHSG) |
| **Metrik** | RMSE, MAPE, R² |

### Data & Variabel

| Jenis Data | Variabel | Frekuensi | Sumber |
|---|---|---|---|
| Pasar modal | IHSG | Harian | Bursa Efek Indonesia (BEI) |
| Makroekonomi | Nilai tukar Rupiah/USD (Kurs) | Harian | Bank Indonesia (BI) |
| Makroekonomi | Inflasi | Bulanan | Badan Pusat Statistik (BPS) |
| Makroekonomi | Suku bunga acuan (BI Rate) | Bulanan | Bank Indonesia (BI) |

- **Split time-series 80:20**, cutoff **2022-12-21** (train 2015-01-02 → 2022-12-20; test 2022-12-21 → 2024-12-30).
- **Scaling** `MinMaxScaler`, *fit hanya pada data train* untuk mencegah kebocoran informasi.

---

## Arsitektur Model

Model utama menggunakan **Dual-Branch LSTM** (Keras Functional API) yang memproses dua kelompok data berfrekuensi berbeda secara terpisah namun simultan:

- **Cabang harian (daily branch):** input IHSG + Kurs (sekuens 90 hari).
- **Cabang bulanan (monthly branch):** input Inflasi + BI Rate (sekuens 3 bulan).

Keluaran kedua cabang **digabungkan pada tingkat representasi** (operasi *concatenate* atas vektor hasil LSTM masing-masing cabang, **bukan** menggabungkan data mentah berfrekuensi berbeda ke dalam satu sekuens), lalu diproses melalui lapisan *Dense* untuk menghasilkan prediksi IHSG t+1. Pemisahan dua cabang dipilih karena perbedaan frekuensi dan panjang sekuens; alternatif *forward-fill* data bulanan ke harian akan membuat nilai berulang ~21 kali (jumlah hari bursa per bulan) dan menimbulkan redundansi.

**Baseline univariate** memakai rancangan setara satu cabang dengan input IHSG saja, agar perbandingan adil (perbedaan hasil hanya disebabkan perbedaan variabel input, bukan kapasitas arsitektur).

### Hyperparameter Final

| Parameter | Nilai |
|---|---|
| `seq_daily` | 90 |
| `seq_monthly` | 3 (dipilih via studi ablasi, RMSE terkecil) |
| `lstm_units` | 128 |
| `dropout` | 0.2 |
| `learning_rate` | 1e-4 |
| `batch_size` | 32 |

Nilai hyperparameter arsitektur dikunci dari observasi empiris pada eksperimen pendahulu (lihat notebook [modeling_multivariate.ipynb](notebook/modeling_multivariate.ipynb) dan [modeling_multivariate_v2.ipynb](notebook/modeling_multivariate_v2.ipynb)).

---

## Hasil (Testing Set)

| Metrik | Univariate | Multivariate |
|---|---|---|
| RMSE | 129.05 | **122.89** |
| MAPE | 1.42% | **1.35%** |
| R² | 0.809 | **0.827** |

Pada studi kasus ini, penambahan variabel makroekonomi memperbaiki metrik titik (RMSE, MAPE, R²) dengan arah yang konsisten. Namun, uji signifikansi statistik **Diebold–Mariano** menunjukkan perbedaan tersebut **belum signifikan pada taraf 5%** (p = 0,082 untuk squared-error loss; 0,063 untuk absolute-error loss) — hanya marginal pada taraf 10%. Jadi keunggulan multivariate bersifat konsisten secara arah, tetapi belum cukup kuat secara statistik untuk diklaim signifikan. Detail uji ada di [notebook/uji_diebold_mariano.ipynb](notebook/uji_diebold_mariano.ipynb).

Figur hasil lengkap (kurva loss, actual vs predicted, residual, distribusi error, rolling MAPE, dll.) tersedia di [reports/figures_v3/](reports/figures_v3/).

---

## Struktur Repositori

```
.
├── notebook/        # Kode: preprocessing, modeling, visualisasi (sumber kebenaran)
├── data/
│   ├── raw/         # Data mentah: IHSG, kurs, inflasi, BI rate
│   └── processed_v2/# Artefak preprocessing final (train/test, scaler, sekuens)
├── model/
│   └── model_multivariate_v3/  # Model & konfigurasi final (multivariate + univariate)
├── reports/
│   └── figures_v3/  # Seluruh figur hasil + summary_table.csv
├── README.md
└── .gitignore
```

---

## Alur Pipeline & Notebook

Jalankan berurutan. Semua path di notebook bersifat relatif dari folder `notebook/`.

### Pipeline final (inti penelitian)

| Tahap | Notebook | Keterangan |
|---|---|---|
| 1. Pra-pemrosesan | [notebook/preprocessing_multivariate_v2.ipynb](notebook/preprocessing_multivariate_v2.ipynb) | Membersihkan & menyelaraskan data harian/bulanan, split train/test, scaling, pembentukan sekuens → menghasilkan artefak di `data/processed_v2/`. |
| 2. Pemodelan & evaluasi | [notebook/modeling_multivariate_v3-2.ipynb](notebook/modeling_multivariate_v3-2.ipynb) | Membangun & melatih model univariate dan multivariate dual-branch, studi ablasi `seq_monthly`, evaluasi RMSE/MAPE/R² → menyimpan ke `model/model_multivariate_v3/`. |
| 3. Visualisasi | [notebook/visualisasi_multivariate_v3-2.ipynb](notebook/visualisasi_multivariate_v3-2.ipynb) | Menghasilkan seluruh figur hasil ke `reports/figures_v3/`. |
| 4. Uji signifikansi | [notebook/uji_diebold_mariano.ipynb](notebook/uji_diebold_mariano.ipynb) | Uji Diebold–Mariano atas galat prediksi tersimpan (post-hoc, tidak melatih ulang) untuk menilai signifikansi statistik selisih akurasi multivariate vs univariate. |

### Eksperimen pendukung (dasar pemilihan hyperparameter)

Notebook berikut **bukan** pipeline final, melainkan eksperimen yang mendasari penguncian hyperparameter pada model final:

| Notebook | Keterangan |
|---|---|
| [notebook/modeling_multivariate.ipynb](notebook/modeling_multivariate.ipynb) | Eksperimen awal (v1): *grid search* hyperparameter. |
| [notebook/modeling_multivariate_v2.ipynb](notebook/modeling_multivariate_v2.ipynb) | Eksperimen lanjutan (v2): uji robustness multi-seed + ablasi awal. |

---

## Cara Menjalankan

Prasyarat: Python 3.10, dengan stack TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib, joblib.

```bash
# 1. Buat environment & install dependensi
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install tensorflow scikit-learn pandas numpy matplotlib joblib jupyter

# 2. Jalankan notebook sesuai urutan pipeline di atas
jupyter notebook
```

Model final dan artefak preprocessing sudah disertakan, sehingga notebook pemodelan/visualisasi dapat langsung dijalankan tanpa mengulang seluruh pelatihan.

---

## Catatan

- Seluruh kode penelitian berada di notebook Jupyter (bukan package Python).
- Repositori ini merupakan artefak penelitian; naskah skripsi dikelola terpisah.
