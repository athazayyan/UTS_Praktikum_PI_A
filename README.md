# INFORMATION RETRIEVAL SYSTEM

Sistem pencarian dokumen berbasis Python menggunakan Whoosh dan Scikit-learn dengan hybrid scoring (BM25 + CountVectorizer + Cosine Similarity).

---

## DAFTAR ISI

1. [Fitur Utama](#fitur-utama)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Instalasi](#instalasi)
5. [Cara Penggunaan](#cara-penggunaan)
6. [Arsitektur Sistem](#arsitektur-sistem)
7. [Evaluasi](#evaluasi)
8. [Troubleshooting](#troubleshooting)

---

## FITUR UTAMA

### Preprocessing
- Text cleaning (lowercase, non-alphanumeric removal)
- Tokenization
- Stopword removal (Sastrawi)
- Stemming (Sastrawi Stemmer)
- Handling individual dataset (removal domain-specific words)

### Information Retrieval
- Whoosh BM25 search engine untuk initial retrieval
- CountVectorizer untuk feature extraction
- Cosine Similarity untuk document ranking
- Multi-field search (title dan content)
- Batch processing untuk indexing dataset besar

### User Interface
- CLI interaktif sederhana
- 3 menu utama (Load/Index, Search, Exit)
- Smart error detection
- User guidance saat salah input

---

## DATASET

### Sumber Data

Sistem ini menggunakan 5 dataset dari berbagai sumber:

| Dataset | Jumlah Dokumen | Sumber | Deskripsi |
|---------|----------------|--------|-----------|
| ETD UGM | 10,000 | etd_ugm.csv | Electronic Thesis and Dissertation Universitas Gadjah Mada |
| ETD USK | 10,000 | etd_usk.csv | Electronic Thesis and Dissertation Universitas Syiah Kuala |
| Kompas | 10,000 | kompas.csv | Artikel berita dari Kompas.com |
| Tempo | 10,000 | tempo.csv | Artikel berita dari Tempo.co |
| Mojok | 9,484 | mojok.csv | Artikel dari Mojok.co |

**Total Dokumen: 49,484**

### Format Dataset

Setiap file CSV memiliki struktur:
```
judul,konten
"Judul dokumen 1","Isi konten dokumen 1"
"Judul dokumen 2","Isi konten dokumen 2"
...
```

### Statistik Dataset Setelah Preprocessing

| Dataset | Total Kata (Konten) | Total Kata (Judul) | Rata-rata per Dokumen |
|---------|---------------------|--------------------|-----------------------|
| UGM | 2,376,071 | 157,212 | 237.6 kata/dokumen |
| USK | 2,114,783 | 160,570 | 211.5 kata/dokumen |
| Mojok | 7,746,514 | 115,646 | 816.8 kata/dokumen |
| Tempo | 4,012,804 | 102,533 | 401.3 kata/dokumen |
| Kompas | 3,129,328 | 100,916 | 312.9 kata/dokumen |

---

## PREPROCESSING

Preprocessing dilakukan untuk membersihkan dan mempersiapkan teks sebelum indexing. Berikut adalah tahapan yang dilakukan:

### 1. Data Loading dan Validasi

```python
import pandas as pd

# Load semua dataset
ugm = pd.read_csv("dataset/etd_ugm.csv")
usk = pd.read_csv("dataset/etd_usk.csv")
kompas = pd.read_csv("dataset/kompas.csv")
mojok = pd.read_csv("dataset/mojok.csv")
tempo = pd.read_csv("dataset/tempo.csv")
```

Setiap dataset divalidasi untuk memastikan memiliki kolom `judul` dan `konten`.

### 2. Text Cleaning

#### a. Normalisasi String
```python
for data in dataset:
    data['judul'] = data['judul'].str.strip()
    data['konten'] = data['konten'].str.strip()
    data['judul'] = data['judul'].astype('string')
    data['konten'] = data['konten'].astype('string')
```

#### b. Lowercase dan Removal Non-Alphanumeric
```python
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Hapus non-alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus whitespace berlebih
    return text
```

Contoh:
```
Input : "KOMPAS.com - Xiaomi mengumumkan kacamata pintar!"
Output: "kompas com xiaomi mengumumkan kacamata pintar"
```

### 3. Handling Domain-Specific Words

Setiap dataset memiliki kata-kata spesifik yang perlu dibersihkan:

#### a. USK Dataset
```python
# Hapus kata "abstrak" yang sering muncul di konten
usk['konten_clean'] = usk['konten_clean'].str.replace(r'\babstrak\b', ' ', regex=True)
```

#### b. Kompas Dataset
```python
# Hapus "kompas.com"
kompas['konten_clean'] = kompas['konten_clean'].str.replace(r'\bkompas\s*com\b', ' ', regex=True)
```

#### c. Tempo Dataset
```python
# Hapus "tempo.co"
tempo['konten_clean'] = tempo['konten_clean'].str.replace(r'\btempo\s*co\b', ' ', regex=True)
```

#### d. Mojok Dataset
```python
# Hapus "mojok.co"
mojok['konten_clean'] = mojok['konten_clean'].str.replace(r'\bmojok\s*co\b', ' ', regex=True)
```

### 4. Tokenization

```python
def tokenize_text(text):
    if pd.isna(text):
        return []
    tokens = text.split()
    return tokens

# Aplikasikan ke semua dataset
for df in dataset_clean:
    df['konten_tokens'] = df['konten'].apply(tokenize_text)
    df['judul_tokens'] = df['judul'].apply(tokenize_text)
```

Contoh:
```
Input : "xiaomi mengumumkan kacamata pintar"
Output: ['xiaomi', 'mengumumkan', 'kacamata', 'pintar']
```

### 5. Stopword Removal

Menggunakan Sastrawi StopWordRemover:

```python
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StopWordRemoverFactory()
global_stopwords = set(factory.get_stop_words())

def remove_stopwords(tokens, stopwords):
    return [word for word in tokens if word not in stopwords]

combined['konten_tokens'] = combined['konten_tokens'].apply(
    lambda x: remove_stopwords(x, global_stopwords)
)
```

Contoh stopwords yang dihapus: "yang", "di", "dari", "dan", "untuk", dll.

Contoh:
```
Input : ['xiaomi', 'mengumumkan', 'kacamata', 'pintar', 'yang', 'baru']
Output: ['xiaomi', 'mengumumkan', 'kacamata', 'pintar', 'baru']
```

### 6. Stemming

Menggunakan Sastrawi Stemmer untuk mengubah kata ke bentuk dasar:

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokens_list(tokens, stemmer):
    if not isinstance(tokens, list):
        return []
    return [stemmer.stem(t) for t in tokens]
```

Contoh:
```
Input : ['mengumumkan', 'berjalan', 'pembelajaran']
Output: ['umum', 'jalan', 'ajar']
```

#### Batch Processing untuk Stemming

Karena dataset besar (49,484 dokumen), stemming dilakukan secara batch dengan checkpoint:

```python
def stem_dataframe_in_chunks(df, chunk_size=1000, save_path='checkpoints_stem'):
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        i0 = chunk_idx * chunk_size
        i1 = min(len(df), (chunk_idx + 1) * chunk_size)
        
        # Stem tokens
        for ridx in range(i0, i1):
            df.at[ridx, 'konten_tokens'] = stem_tokens_list(
                df.at[ridx, 'konten_tokens'], stemmer
            )
            df.at[ridx, 'judul_tokens'] = stem_tokens_list(
                df.at[ridx, 'judul_tokens'], stemmer
            )
        
        # Save checkpoint
        save_checkpoint(df.iloc[i0:i1], chunk_idx, save_path)
```

### 7. Dataset Combination

Setelah preprocessing, semua dataset digabung:

```python
combined = pd.concat([ugm, usk, mojok, tempo, kompas], ignore_index=True)
```

Kolom akhir:
- `konten_tokens`: List token dari konten (sudah di-stem)
- `judul_tokens`: List token dari judul (sudah di-stem)
- `konten_len`: Jumlah token di konten
- `judul_len`: Jumlah token di judul
- `source`: Label sumber dataset

### 8. Export untuk Indexing

Dataset akhir disimpan dalam format CSV:

```python
combined.to_csv('dataset/combined_stemmed_dataset.csv', index=False)
```

Format file:
```csv
konten_tokens,judul_tokens,konten_len,judul_len,source
"['drone', 'umum', 'klasifikasi', ...]","['tingkat', 'guna', ...]",173,12,UGM
...
```

### Ringkasan Pipeline Preprocessing

```
Raw Dataset
    |
    v
[1] Load & Validation
    |
    v
[2] Text Cleaning (lowercase, remove non-alphanumeric)
    |
    v
[3] Domain-Specific Cleaning (remove "kompas.com", "tempo.co", etc)
    |
    v
[4] Tokenization (split by whitespace)
    |
    v
[5] Stopword Removal (Sastrawi)
    |
    v
[6] Stemming (Sastrawi, batch processing)
    |
    v
[7] Combine All Datasets
    |
    v
[8] Export to CSV
    |
    v
Ready for Indexing
```

### Waktu Preprocessing

Estimasi waktu untuk 49,484 dokumen:
- Text cleaning: ~2 menit
- Tokenization: ~1 menit
- Stopword removal: ~3 menit
- Stemming: ~5 jam (dengan batch processing)
- Total: ~5 jam 6 menit

### File Output Preprocessing

```
dataset/
├── combined_stemmed_dataset.csv    # Dataset gabungan siap index
└── checkpoints_stem/               # Checkpoint stemming
    ├── stem_checkpoint_chunk_0000.pkl
    ├── stem_checkpoint_chunk_0001.pkl
    └── ...
```

---

## INSTALASI

### 1. Requirements

Python 3.7 atau lebih tinggi

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies yang dibutuhkan:
```
pandas>=1.5.0
whoosh>=2.7.4
scikit-learn>=1.0.0
tqdm>=4.64.0
Sastrawi>=1.0.1
```

### 3. Struktur Folder

```
information-retrieval-system/
├── dataset/
│   └── combined_stemmed_dataset.csv
├── modules/
│   ├── __init__.py
│   ├── indexer.py
│   └── searcher.py
├── main.py
├── requirements.txt
└── README.md
```

### 4. Persiapan Dataset

Pastikan file `combined_stemmed_dataset.csv` sudah ada di folder `dataset/`.

Format file harus memiliki kolom:
- `judul_tokens`: List token dari judul (format string: "['token1', 'token2']")
- `konten_tokens`: List token dari konten (format string: "['token3', 'token4']")
- `source`: Sumber dokumen (contoh: "UGM", "USK", "Kompas", dll)

---

## CARA PENGGUNAAN

### Jalankan Program

```bash
python main.py
```

### Menu Interface

```
=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================
Pilih menu (ketik angka 1/2/3):
```

### Menu 1: Load & Index Dataset

Membuat index dari dataset untuk pertama kali.

**Langkah:**
1. Pilih menu [1]
2. Jika index sudah ada, sistem akan menanyakan apakah ingin rebuild
3. Tunggu proses indexing selesai (progress bar akan muncul)

**Output:**
```
Membuat index pencarian ...

Dataset ditemukan! Jumlah dokumen: 49484

Indexing dokumen: 100%|████████| 49484/49484 [02:30<00:00, 328.12doc/s]

Indexing selesai!
   Berhasil: 49484 dokumen
```

**Catatan:**
- Index hanya perlu dibuat sekali
- Rebuild hanya jika dataset berubah
- Waktu indexing: ~2-3 menit untuk 49,484 dokumen

### Menu 2: Search Query

Mencari dokumen berdasarkan keyword.

**Langkah:**
1. Pilih menu [2]
2. Masukkan query pencarian
3. Sistem akan menampilkan top 5 dokumen relevan

**Contoh:**

```
Pilih menu (ketik angka 1/2/3): 2

============================================================
MODE PENCARIAN
============================================================
Query: pembelajaran online

------------------------------------------------------------
Mencari: 'pembelajaran online'...
------------------------------------------------------------
Ditemukan 5 dokumen relevan!

============================================================
HASIL PENCARIAN (Top 5)
============================================================

[1] Relevance Score: 0.8523
────────────────────────────────────────────────────────────
Judul : Pengaruh Pembelajaran Online Terhadap Motivasi Belajar
Sumber: UGM
Konten: Penelitian ini membahas dampak pembelajaran online terhadap motivasi...
────────────────────────────────────────────────────────────

[2] Relevance Score: 0.7845
────────────────────────────────────────────────────────────
Judul : Efektivitas E-Learning di Masa Pandemi
Sumber: USK
Konten: Pandemi COVID-19 memaksa institusi pendidikan beralih ke pembelajaran...
────────────────────────────────────────────────────────────

[...]

Tips: Gunakan query lebih spesifik untuk hasil lebih akurat

Tekan Enter untuk kembali ke menu...
```

### Menu 3: Exit

Keluar dari program.

---

## ARSITEKTUR SISTEM

### 1. Indexing Process

```
Dataset CSV
    |
    v
Parse Tokens (ast.literal_eval)
    |
    v
Whoosh Schema Creation
    |
    v
Batch Processing (commit per 1000 docs)
    |
    v
Index Storage
```

**File: modules/indexer.py**

Fungsi utama:
- `create_search_index()`: Membuat index dari dataset
- `safe_parse_tokens()`: Parse token string dengan error handling
- Batch commit setiap 1000 dokumen untuk efisiensi

### 2. Search Process

```
User Query
    |
    v
Whoosh Query Parser (multi-field: title + content)
    |
    v
BM25 Search (initial retrieval)
    |
    v
CountVectorizer Feature Extraction
    |
    v
Cosine Similarity Calculation
    |
    v
Ranking & Top K Selection
    |
    v
Display Results
```

**File: modules/searcher.py**

Komponen:
1. **Whoosh BM25**: Initial retrieval dari index
2. **CountVectorizer**: Membuat term frequency vectors
3. **Cosine Similarity**: Menghitung kesamaan antara query dan dokumen

Formula:
```
CountVector(word) = frequency of word in document

Cosine Similarity = 
    dot(vector_query, vector_doc) / 
    (norm(vector_query) × norm(vector_doc))
```

### 3. Scoring Method

#### Whoosh BM25 (Initial Retrieval)

BM25 (Best Matching 25) adalah probabilistic ranking function:

```
score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
```

Dimana:
- `f(qi,D)`: frekuensi term qi dalam dokumen D
- `|D|`: panjang dokumen D
- `avgdl`: rata-rata panjang dokumen
- `k1`, `b`: parameter tuning (default: k1=1.2, b=0.75)

#### CountVectorizer + Cosine Similarity (Reranking)

Setelah initial retrieval dengan BM25, hasil di-rerank dengan:

1. **CountVectorizer**: Membuat matrix term frequency
```
Document: "machine learning adalah pembelajaran mesin"
Vector: [machine:1, learning:1, adalah:1, pembelajaran:1, mesin:1]
```

2. **Cosine Similarity**: Menghitung sudut antara vector query dan dokumen
```
similarity = cos(θ) = (A·B) / (||A|| × ||B||)
```

Range: 0.0 (tidak mirip) sampai 1.0 (sangat mirip)

### 4. Multi-Field Search

Pencarian dilakukan pada 2 field:
- **title**: Judul dokumen (bobot lebih tinggi)
- **content**: Konten dokumen

```python
parser = MultifieldParser(
    ["title", "content"], 
    schema,
    group=OrGroup  # OR logic: cari di title ATAU content
)
```

---

## EVALUASI

### Performance Metrics

Untuk 49,484 dokumen:

| Metrik | Nilai |
|--------|-------|
| Indexing Time | ~2-3 menit |
| Search Time (per query) | <0.5 detik |
| Index Size | ~150 MB |
| Memory Usage | ~500 MB (saat indexing) |

### Search Quality

Sistem menggunakan CountVectorizer + Cosine Similarity yang memberikan:
- Exact keyword matching
- Frequency-based ranking
- Simple dan mudah dipahami

**Kelebihan:**
- Fast computation
- Interpretable results
- Good untuk exact term matching

**Limitasi:**
- Kata umum mendapat bobot tinggi
- Tidak membedakan kata penting vs tidak penting
- Tidak capture semantic similarity

### Contoh Query Performance

| Query | Jumlah Hasil | Waktu | Top 1 Relevance Score |
|-------|--------------|-------|----------------------|
| "pembelajaran online" | 1,234 | 0.3s | 0.85 |
| "covid 19" | 4,567 | 0.4s | 0.92 |
| "kecerdasan buatan" | 892 | 0.2s | 0.78 |
| "ekonomi digital" | 2,103 | 0.3s | 0.81 |

---

## TROUBLESHOOTING

### Problem 1: "Index belum dibuat"

**Penyebab:** Belum menjalankan menu [1] Load & Index Dataset

**Solusi:**
```bash
python main.py
# Pilih menu [1] terlebih dahulu
```

### Problem 2: "Dataset tidak ditemukan"

**Penyebab:** File `combined_stemmed_dataset.csv` tidak ada di folder `dataset/`

**Solusi:**
```bash
# Pastikan struktur folder benar
ls dataset/combined_stemmed_dataset.csv

# Jika tidak ada, jalankan preprocessing terlebih dahulu
```

### Problem 3: "Error parsing tokens"

**Penyebab:** Format kolom token tidak sesuai

**Solusi:**

Format yang benar:
```csv
"['token1', 'token2', 'token3']"
```

Format yang salah:
```csv
['token1', 'token2', 'token3']  # Tanpa quotes
```

### Problem 4: Search lambat

**Penyebab:** Dataset terlalu besar atau index tidak optimal

**Solusi:**
- Kurangi `top_k` (default 5)
- Rebuild index
- Gunakan SSD untuk storage

### Problem 5: Hasil tidak relevan

**Penyebab:** Query terlalu umum atau ambigu

**Solusi:**
- Gunakan query lebih spesifik (3-5 kata)
- Tambahkan context pada query
- Contoh:
  - Kurang baik: "belajar"
  - Lebih baik: "metode pembelajaran online efektif"

### Problem 6: User salah input query di menu

**Penyebab:** User memasukkan query saat diminta pilih menu

**Contoh:**
```
Pilih menu (ketik angka 1/2/3): pembelajaran online
```

**Solusi:**

Sistem akan memberikan guidance:
```
Sepertinya Anda memasukkan query, bukan pilihan menu.
Cara yang benar:
   1. Ketik angka '2' lalu Enter
   2. Kemudian masukkan query Anda
```

### Problem 7: Memory Error saat indexing

**Penyebab:** Dataset terlalu besar untuk RAM yang tersedia

**Solusi:**
- Sistem sudah menggunakan batch processing (commit per 1000 docs)
- Tutup aplikasi lain saat indexing
- Minimum RAM: 4GB (recommended: 8GB)

---

## REFERENSI

### Algoritma dan Metode

1. **BM25 (Best Matching 25)**
   - Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.

2. **CountVectorizer**
   - Scikit-learn documentation: Feature extraction from text
   
3. **Cosine Similarity**
   - Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.

### Libraries

- **Whoosh**: Python search engine library
- **Scikit-learn**: Machine learning library untuk Python
- **Sastrawi**: Indonesian text processing library
- **Pandas**: Data manipulation library
- **tqdm**: Progress bar library

---


## CHANGELOG

### Version 2.0 (Current)
- Preprocessing pipeline lengkap (cleaning, tokenization, stemming)
- CountVectorizer + Cosine Similarity untuk ranking
- Multi-field search (title + content)
- Batch processing untuk indexing
- Smart error detection dan user guidance
- Improved UI/UX

### Version 1.0 (Initial)
- Basic Whoosh search
- Simple text matching
- Single-field search

---

Last Updated: October 2025
