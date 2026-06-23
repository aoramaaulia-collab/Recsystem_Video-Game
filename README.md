# Sistem Rekomendasi Video Game

> Sistem rekomendasi produk e-commerce menggunakan dataset **Amazon Reviews 2023 (Video Games)**. Proyek ini membangun dan membandingkan tiga metode rekomendasi (Content-Based, Item-Based Collaborative, dan SVD) sebagai sebuah eksperimen, lalu menerapkan metode terbaik ke dalam aplikasi web interaktif dengan Streamlit.

**Penjelasan lengkap (artikel):** [Baca di Medium](TEMPEL_LINK_MEDIUM_DI_SINI)

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Tujuan Proyek](#tujuan-proyek)
- [Dataset](#dataset)
- [Fitur Aplikasi](#fitur-aplikasi)
- [Metode yang Dibandingkan](#metode-yang-dibandingkan)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Insight Utama](#insight-utama)
- [Tech Stack](#tech-stack)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan Aplikasi](#cara-menjalankan-aplikasi)
- [Metodologi Singkat](#metodologi-singkat)
- [Author](#author)

---

## Latar Belakang

Platform e-commerce memiliki katalog produk yang sangat besar. Pada kategori video game Amazon saja, terdapat puluhan ribu produk berbeda. Skala ini menimbulkan dua masalah sekaligus. Dari sisi pengguna, mereka kesulitan menemukan produk yang sesuai selera di tengah banyaknya pilihan. Dari sisi bisnis, platform kehilangan potensi penjualan ketika produk yang relevan tidak pernah terlihat oleh calon pembeli yang tepat.

Sistem rekomendasi menjawab masalah ini dengan menyajikan produk yang tepat kepada pengguna yang tepat secara otomatis. Dalam e-commerce, sistem rekomendasi adalah salah satu penggerak utama keterlibatan pengguna, konversi penjualan, dan retensi pelanggan.

## Tujuan Proyek

Proyek ini menjawab beberapa pertanyaan utama:

- Bagaimana membantu pengguna menemukan game yang relevan tanpa harus menelusuri ribuan produk secara manual?
- Apakah personalisasi memberi nilai lebih dibanding sekadar menampilkan produk terpopuler kepada semua orang?
- Dari beberapa metode rekomendasi, mana yang paling cocok untuk karakteristik data ini?
- Bagaimana mengukur secara objektif apakah sebuah sistem rekomendasi benar-benar bekerja dengan baik?

Untuk menjawabnya, proyek ini dikemas sebagai sebuah eksperimen: tiga metode dibangun dan dievaluasi secara adil dengan metrik yang sesuai, lalu metode pemenang diterapkan ke aplikasi.

## Dataset

Proyek menggunakan dataset **Amazon Reviews 2023** (McAuley Lab, UCSD), kategori **Video Games**, yang terdiri dari dua berkas yang digabungkan berdasarkan ID produk (`parent_asin`):

- **Data Reviews (ulasan):** catatan rating pengguna terhadap produk (skala 1 sampai 5) beserta waktunya. Ukuran mentah sekitar 4,6 juta baris.
- **Data Metadata (produk):** informasi produk seperti judul, kategori, deskripsi, dan fitur. Ukuran mentah sekitar 137 ribu baris.

Setelah penggabungan, pembersihan, dan penyaringan, data akhir yang dipakai untuk pemodelan berisi 81.447 interaksi dari 9.803 pengguna dan 5.023 produk.

## Fitur Aplikasi

Aplikasi Streamlit memiliki tiga tab, masing-masing didukung metode yang berbeda:

- **Cari yang Mirip (Content-Based).** Pengguna memilih kategori dan judul game, lalu menerima rekomendasi game lain yang kontennya mirip (genre, deskripsi, fitur serupa).
- **Buat Kamu (Item-Based Collaborative).** Pengguna memilih beberapa game favorit, lalu menerima rekomendasi personal berdasarkan pola perilaku pengguna lain yang serupa. Metode ini adalah pemenang eksperimen.
- **Lagi Hits (Popularitas).** Menampilkan game terpopuler berdasarkan jumlah ulasan, dapat disaring berdasarkan kategori dan rating 4 ke atas untuk menemukan game berkualitas.

Setiap rekomendasi dapat diklik untuk menampilkan detail produk seperti deskripsi.

## Metode yang Dibandingkan

Tiga pendekatan dibangun dan diuji:

- **Content-Based Filtering.** Mengubah teks produk (judul, kategori, deskripsi) menjadi vektor angka dengan TF-IDF, lalu menghitung kemiripan antar produk dengan cosine similarity. Merekomendasikan produk yang kontennya mirip dengan yang disukai pengguna.
- **Item-Based Collaborative Filtering.** Membangun matriks user-item dari rating, lalu menghitung kemiripan antar produk berdasarkan pola siapa yang menyukainya. Merekomendasikan produk berdasarkan kebiasaan "yang menyukai X juga menyukai Y".
- **SVD (Model-Based / Matrix Factorization).** Memecah matriks rating menjadi faktor laten untuk memprediksi rating. Diperbaiki dengan koreksi bias (rata-rata global, bias pengguna, bias produk) agar prediksi berada pada skala yang benar.

## Hasil Eksperimen

Evaluasi dilakukan dengan menyembunyikan sebagian rating pengguna, lalu memeriksa apakah produk yang disembunyikan muncul di rekomendasi. Metrik yang dipakai: Precision@K dan Hit Rate untuk semua metode, serta RMSE dan MAE untuk SVD.

| Metode | Precision@5 | Hit Rate@5 | Precision@10 | Hit Rate@10 | Precision@20 | Hit Rate@20 |
|--------|:-----------:|:----------:|:------------:|:-----------:|:------------:|:-----------:|
| **Item-Based Collaborative** | **0,0264** | **0,1207** | **0,0202** | **0,1779** | **0,0154** | **0,2568** |
| SVD (Model-Based) | 0,0149 | 0,0705 | 0,0104 | 0,0956 | 0,0069 | 0,1237 |
| Baseline Populer | 0,0073 | 0,0360 | 0,0069 | 0,0651 | 0,0058 | 0,1094 |

**Kesimpulan:** Item-Based Collaborative Filtering unggul di semua nilai K, sekitar 3 kali lipat baseline populer dan sekitar 2 kali lipat SVD. Untuk prediksi rating, SVD diperbaiki dengan koreksi bias sehingga RMSE turun dari 4,68 menjadi 0,77 (MAE 0,495).

## Insight Utama

- **Personalisasi terbukti bekerja.** Item-based mengalahkan baseline populer sekitar 3 kali lipat, membuktikan personalisasi memberi nilai nyata dibanding sekadar menampilkan produk populer.
- **Yang sederhana mengalahkan yang kompleks.** Item-based collaborative mengungguli SVD yang lebih canggih. Pada data e-commerce yang sangat jarang (sparse), mengukur kemiripan produk secara langsung lebih andal daripada mempelajari faktor laten.
- **Akurasi prediksi tidak sama dengan kualitas rekomendasi.** SVD memiliki RMSE yang baik (0,77) tetapi Precision@K yang lebih rendah. Menebak nilai rating dengan tepat berbeda dari menempatkan produk yang relevan di peringkat teratas.
- **Bahaya kebocoran data (data leakage).** Evaluasi awal sempat menghasilkan nilai nol karena produk uji masih ikut terpakai saat menghitung rekomendasi. Diperbaiki dengan membangun matriks training yang benar-benar terpisah dari data uji.

## Tech Stack

- **Bahasa:** Python
- **Pengolahan data:** pandas, NumPy
- **Machine Learning:** scikit-learn (TF-IDF, Cosine Similarity, TruncatedSVD)
- **Data warehouse:** Google BigQuery (ETL data berukuran jutaan baris)
- **Visualisasi:** Matplotlib, Seaborn
- **Aplikasi web:** Streamlit

## Struktur Proyek

```
Recsystem_Video-Game/
├── Deployment/
│   ├── app.py                          # Aplikasi Streamlit
│   ├── model_artifacts.pkl             # Artefak model (TF-IDF + matriks kemiripan)
│   ├── video_games_items.parquet       # Data produk
│   └── video_games_popularitas.parquet # Data produk populer
├── notebooks/
│   ├── ETL.ipynb                       # Extract, Transform, Load ke BigQuery
│   └── Analisis_dan_Modelling.ipynb    # EDA + pemodelan + evaluasi
├── requirements.txt
└── README.md
```

> **Catatan:** file data mentah dan parquet berukuran besar (ratusan MB) tidak disertakan karena melebihi batas ukuran file GitHub (100 MB). Data dapat diunduh dari sumbernya (Amazon Reviews 2023, McAuley Lab UCSD) lalu diproses ulang dengan notebook ETL. Sesuaikan nama berkas pada struktur di atas dengan nama berkas di repo kamu jika berbeda.

## Cara Menjalankan Aplikasi

```bash
# 1. Clone repository
git clone https://github.com/aoramaaulia-collab/Recsystem_Video-Game.git
cd Recsystem_Video-Game

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan aplikasi dari folder Deployment
cd Deployment
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`. Pastikan ketiga artefak (`model_artifacts.pkl`, `video_games_items.parquet`, `video_games_popularitas.parquet`) berada di folder yang sama dengan `app.py`, karena aplikasi memuat artefak tersebut, bukan membangun ulang model.

Aplikasi yang sudah dideployment ke streamlit.io juga dapat dilihat pada link berikut: https://recsystemvideo-game-fpjzm9twfhdh4uer22bgyb.streamlit.app/

## Metodologi Singkat

```
Data Mentah → ETL & k-core (BigQuery) → Pembersihan → EDA → Sampling
→ Content-Based + Item-Based + SVD → Evaluasi & Perbandingan → Deployment
```

1. **ETL:** pengambilan dan penggabungan data di BigQuery, penyaringan k-core (minimal 5 interaksi per pengguna dan produk).
2. **Pembersihan dan EDA:** penanganan duplikat, analisis sparsity, pola long-tail, dan tren, melalui delapan visualisasi.
3. **Sampling:** 20.000 pengguna, menghasilkan 81.447 interaksi (9.803 pengguna, 5.023 produk) agar muat di memori.
4. **Pemodelan:** tiga metode rekomendasi dibangun (Content-Based, Item-Based, SVD).
5. **Evaluasi:** Precision@K, Hit Rate, RMSE, dan MAE, dengan penanganan kebocoran data (data leakage).
6. **Deployment:** metode pemenang (Item-Based) diterapkan ke aplikasi Streamlit, dengan menjaga konsistensi antara model yang diuji dan yang dipakai aplikasi.

## Author

**Aulia Aorama**
[Lihat Lebih lanjut Portofolio saya](https://auliaaorama-porto.netlify.app/)
