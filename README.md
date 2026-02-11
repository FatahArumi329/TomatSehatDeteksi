# TomatAI: Sistem Deteksi Penyakit Daun Tomat Menggunakan MobileNetV2

TomatAI adalah aplikasi berbasis web yang diimplementasikan menggunakan **Streamlit** untuk mengklasifikasikan penyakit pada tanaman tomat berdasarkan citra daun. Proyek ini menggunakan arsitektur **Deep Learning MobileNetV2** yang dioptimalkan untuk efisiensi komputasi tanpa mengorbankan performa model secara signifikan.

**Akses Aplikasi:** [deteksi-tomat-sht.streamlit.app](https://deteksi-tomat-sht.streamlit.app/)

## 📝 Latar Belakang

Identifikasi penyakit hawar daun (*blight*) pada tanaman tomat sering kali menjadi tantangan bagi petani karena kemiripan gejala visualnya. Proyek ini bertujuan untuk menyediakan alat diagnosa otomatis yang akurat dengan memanfaatkan teknik *Transfer Learning* dan *Fine-Tuning* pada dataset citra daun tomat guna meminimalisir risiko gagal panen.

## 🛠️ Metodologi Pengembangan Model

Model dibangun menggunakan skrip `pipeline.py` dengan tahapan sebagai berikut:

1. **Arsitektur Base Model:** Menggunakan **MobileNetV2** dengan bobot praterlatih (*pre-trained weights*) dari ImageNet.
2. **Feature Extraction:** Seluruh layer pada *base model* dibekukan (*frozen*), kemudian dilatih lapisan klasifikasi baru (Dense Layer 128) dengan *learning rate*  selama 10 epoch.
3. **Fine-Tuning:** Membuka blokir lapisan (*unfreezing*) mulai dari index layer ke-120 ke atas. Proses ini menggunakan *learning rate* yang lebih rendah () selama 10 epoch untuk menyesuaikan bobot model dengan tekstur spesifik penyakit tomat.
4. **Optimasi:** Menggunakan optimizer *Adam* dan fungsi kerugian *Categorical Crossentropy*.

## 🚀 Fitur Aplikasi

* **Prediksi Real-Time:** Unggah citra daun (JPG/PNG) untuk mendapatkan diagnosa instan.
* **Visualisasi Probabilitas:** Grafik batang interaktif menggunakan *Plotly* yang menunjukkan tingkat kepercayaan model terhadap setiap kelas.
* **Log Riwayat:** Penyimpanan data hasil pemeriksaan dalam sesi berjalan yang dapat diekspor ke format CSV.
* **Kamus Penyakit:** Informasi komprehensif mengenai ciri fisik penyakit serta rekomendasi solusi penanganannya.

## 📊 Hasil dan Performa

Berdasarkan pengujian pada data validasi (120 citra), model mencapai akurasi keseluruhan sebesar **88%**. Berikut adalah rincian performa per kategori:

```markdown
| Kelas | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato Early Blight** | 0.84 | 0.78 | 0.81 | 40 |
| **Tomato Late Blight** | 0.97 | 0.85 | 0.91 | 40 |
| **Tomato Healthy** | 0.83 | 1.00 | 0.91 | 40 |
| **Akurasi Keseluruhan** | | | **0.88** | **120** |

```

**Confusion Matrix:**

```text
[[31  1  8]  <- Early Blight
 [ 6 34  0]  <- Late Blight
 [ 0  0 40]] <- Healthy

```

Tabel Dependensi

Jika ingin menambahkan spesifikasi teknologi yang digunakan secara formal:

```markdown
| Library | Versi | Deskripsi |
| :--- | :--- | :--- |
| **TensorFlow** | 2.15.0+ | Framework utama Deep Learning |
| **Streamlit** | Terbaru | Framework deployment web |
| **NumPy** | Terbaru | Komputasi matriks citra |
| **Plotly** | Terbaru | Visualisasi interaktif |

```

## 👥 Tim Pengembang (Kelompok 3)

* Achmad Karis Wibowo
* Albert Cendra Hermawan
* Yosia Marpaung
* Dhimas Muhammad Fattah Arrumy


## 📚 Daftar Pustaka

1. Dokumentasi Resmi TensorFlow, Keras, dan Streamlit.

