"""
Aplikasi Rekomendasi Video Game
================================
Content-Based Filtering + Item-Based Collaborative Filtering
Data: Amazon Reviews 2023 (kategori Video Games)

Alur dibuat bertahap: pilih kategori -> cari & pilih judul (tersaring kategori)
-> rekomendasi muncul. Aplikasi memuat artefak model yang sama persis dengan
yang diuji di notebook (konsistensi train/serve). Filter, tampilan detail, dan
info rating hanyalah lapisan navigasi/tampilan; input ke model tetap sama
(game yang dipilih). Rating dan jumlah ulasan diambil dari data popularitas
hanya sebagai informasi, tidak memengaruhi skor model.

Cara menjalankan (dari folder yang berisi app.py + file artefak):
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import html
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(page_title="Rekomendasi Game", page_icon="🎮", layout="wide")
BASE = Path(__file__).parent

# ----------------------------------------------------------------------
# Gaya tampilan (CSS)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #7C3AED 0%, #C026D3 55%, #EC4899 100%);
        border-radius: 22px; padding: 38px 32px; color: #fff;
        box-shadow: 0 12px 32px rgba(124,58,237,0.28); margin-bottom: 20px;
    }
    .hero h1 { font-size: 36px; font-weight: 800; margin: 0; color: #fff; }
    .hero p  { font-size: 16px; opacity: .95; margin: 9px 0 0 0; }
    .selected {
        background: linear-gradient(135deg, #F5EEFF 0%, #FBEFF8 100%);
        border-left: 6px solid #7C3AED; padding: 15px 18px;
        border-radius: 12px; margin: 12px 0 8px 0; color: #1F1B2E; font-size: 16px;
        box-shadow: 0 4px 14px rgba(124,58,237,0.10);
    }
    .selected span { color: #6D5BA6; font-size: 13px; }
    .badge {
        display: inline-block; font-size: 12px; color: #6D5BA6;
        background: linear-gradient(135deg, #F1ECFC 0%, #FBEAF5 100%);
        padding: 4px 12px; border-radius: 20px; margin-bottom: 6px;
    }
    /* Judul kecil di dalam kartu (Deskripsi / Fitur) */
    .kartu-label {
        font-size: 12px; font-weight: 700; letter-spacing: .4px;
        text-transform: uppercase; color: #7C3AED !important; margin: 10px 0 2px 0;
    }
    /* Isi teks di dalam kartu (deskripsi, fitur) selalu gelap & terbaca */
    .kartu-teks { color: #2A2440 !important; font-size: 15px; line-height: 1.6; margin: 2px 0; }
    /* Baris rating di dalam kartu */
    .kartu-rating { font-size: 15px; font-weight: 600; color: #2A2440 !important; margin: 6px 0; }
    .kartu-rating .bintang { color: #F59E0B !important; }
    .kartu-rating .ulasan  { color: #6D5BA6 !important; font-weight: 500; }

    /* --- Kartu rekomendasi (expander): tepi ungu, isi putih, gradasi lembut --- */
    [data-testid="stExpander"] {
        border: none !important;
        border-radius: 14px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F1FF 100%);
        border-left: 5px solid #7C3AED !important;
        box-shadow: 0 4px 14px rgba(124,58,237,0.10);
        overflow: hidden;
    }
    /* Header kartu (judul game): latar putih + teks gelap di SEMUA keadaan
       (tertutup, TERBUKA, disorot). Saat terbuka, Streamlit memberi header
       latar gelap; kita paksa transparan agar latar putih kartu yang terlihat.
       -webkit-text-fill-color dipakai karena warna teks kadang diset lewat
       properti itu sehingga 'color' saja tidak cukup. */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details[open] > summary {
        background: transparent !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div,
    [data-testid="stExpander"] details[open] summary,
    [data-testid="stExpander"] details[open] summary p,
    [data-testid="stExpander"] details[open] summary span,
    [data-testid="stExpander"] details[open] summary div {
        font-weight: 600;
        color: #2A2440 !important;
        -webkit-text-fill-color: #2A2440 !important;
    }
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover p {
        color: #7C3AED !important;
        -webkit-text-fill-color: #7C3AED !important;
    }
    [data-testid="stExpander"] summary svg { fill: #7C3AED !important; }
    /* Latar isi kartu tetap putih bergradasi + semua teks default jadi gelap.
       Ini jaring pengaman bila ada teks yang tidak memakai kelas khusus. */
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F1FF 100%);
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] label,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] [data-testid="stProgress"] * {
        color: #2A2440 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Muat artefak model (cache)
# ----------------------------------------------------------------------
@st.cache_resource
def muat_artefak():
    with open(BASE / "model_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    items = pd.read_parquet(BASE / "video_games_items.parquet").reset_index(drop=True)
    populer = pd.read_parquet(BASE / "video_games_popularitas.parquet").reset_index(drop=True)
    return art, items, populer


try:
    art, items, populer = muat_artefak()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan. Pastikan ketiga file ini ada di folder yang "
        "sama dengan app.py:\n\n- model_artifacts.pkl\n- video_games_items.parquet\n"
        "- video_games_popularitas.parquet"
    )
    st.stop()

tfidf_matrix    = art["tfidf_matrix"]
item_similarity = art["item_similarity"]
item_ke_idx     = art["item_ke_idx"]
idx_ke_item     = art["idx_ke_item"]

# Peta rating agregat (dari data populer). Ini INFORMASI TAMPILAN saja, bukan
# input model, sehingga konsistensi train/serve tetap terjaga.
PETA_RATING = populer.set_index("parent_asin")[["rata_rating", "jumlah_rating"]]


def tempel_rating(df):
    """Menambahkan kolom rata_rating & jumlah_rating berdasarkan parent_asin."""
    if "parent_asin" in df.columns:
        df = df.merge(PETA_RATING, on="parent_asin", how="left")
    return df


# ----------------------------------------------------------------------
# Kategori sederhana (platform) untuk filter
# ----------------------------------------------------------------------
def platform_dari_kategori(kat):
    if pd.isna(kat):
        return "Lainnya"
    k = str(kat).lower()
    if "playstation" in k:
        return "PlayStation"
    if "nintendo" in k or "wii" in k or "gamecube" in k or "switch" in k:
        return "Nintendo"
    if "xbox" in k:
        return "Xbox"
    if "sega" in k or "dreamcast" in k or "genesis" in k or "saturn" in k:
        return "Sega"
    if "pc" in k:
        return "PC"
    if "mac" in k:
        return "Mac"
    return "Lainnya"


items["platform"] = items["categories"].apply(platform_dari_kategori)
DAFTAR_PLATFORM = ["Semua"] + sorted(items["platform"].dropna().unique().tolist())


def rapikan(teks):
    """Merapikan teks deskripsi: buang format list, awalan 'Product Description',
    dan rapikan spasi berlebih. Hanya memengaruhi tampilan, bukan data model."""
    s = str(teks).strip()
    if s.lower() in ("", "nan", "none", "[]"):
        return ""
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace("', '", " · ").replace('", "', " · ")
    s = s.strip().strip("'\"").strip()
    # Buang awalan editorial yang menempel di kalimat
    for awalan in ("Product Description", "Product description"):
        if s.startswith(awalan):
            s = s[len(awalan):].lstrip(" :.-")
            break
    s = " ".join(s.split())  # rapikan spasi/tab/newline berlebih
    return s.strip()


# ----------------------------------------------------------------------
# Fungsi rekomendasi (memakai artefak yang sama dengan model teruji)
# ----------------------------------------------------------------------
KOLOM_TAMPIL = ["product_title", "categories", "description", "features", "platform"]


@st.cache_data
def produk_serupa(pos_item: int, top_n: int = 10):
    vektor = tfidf_matrix[pos_item]
    kemiripan = cosine_similarity(vektor, tfidf_matrix).flatten()
    kemiripan[pos_item] = -1.0
    urut = kemiripan.argsort()[::-1][:top_n]
    # Sertakan parent_asin agar bisa ditempeli rating (info tampilan).
    kolom = [c for c in (["parent_asin"] + KOLOM_TAMPIL) if c in items.columns]
    hasil = items.iloc[urut][kolom].copy()
    hasil["skor"] = kemiripan[urut]
    hasil = tempel_rating(hasil)
    return hasil.reset_index(drop=True)


def mungkin_suka(pos_disuka, top_n: int = 10):
    rating_vec = np.zeros(len(item_ke_idx))
    dipilih = []
    for pos in pos_disuka:
        asin = items.loc[pos, "parent_asin"]
        if asin in item_ke_idx:
            j = item_ke_idx[asin]
            rating_vec[j] = 5.0
            dipilih.append(j)
    skor = np.asarray(item_similarity.dot(rating_vec)).flatten()
    for j in dipilih:
        skor[j] = -1.0
    urut = np.argsort(skor)[::-1][:top_n]
    asin_hasil = [idx_ke_item[i] for i in urut]
    peta = {idx_ke_item[i]: skor[i] for i in urut}
    kolom = [c for c in (["parent_asin"] + KOLOM_TAMPIL) if c in items.columns]
    hasil = items[items["parent_asin"].isin(asin_hasil)][kolom].copy()
    hasil["skor"] = hasil["parent_asin"].map(peta)
    hasil = tempel_rating(hasil)
    return hasil.sort_values("skor", ascending=False).reset_index(drop=True)


def rekomendasi_populer(top_n=10, min_rating=None, platform=None):
    base = populer.copy()
    if min_rating is not None and "rata_rating" in base.columns:
        base = base[base["rata_rating"] >= min_rating]
    butuh = [c for c in ["product_title", "categories", "description", "features"]
             if c not in base.columns]
    if butuh and "parent_asin" in base.columns:
        base = base.merge(items[["parent_asin"] + butuh], on="parent_asin", how="left")
    if "platform" not in base.columns and "categories" in base.columns:
        base["platform"] = base["categories"].apply(platform_dari_kategori)
    if "jumlah_rating" in base.columns:
        base = base.sort_values("jumlah_rating", ascending=False)
    if platform is not None and platform != "Semua" and "platform" in base.columns:
        base = base[base["platform"] == platform]
    return base.head(top_n)


# ----------------------------------------------------------------------
# Tampilan daftar game: tiap game bisa diklik untuk melihat detail
# ----------------------------------------------------------------------
def _baris_rating_html(row):
    """Membuat HTML baris rating + jumlah ulasan bila datanya ada."""
    rr, jr = row.get("rata_rating"), row.get("jumlah_rating")
    bagian = []
    if pd.notna(rr):
        bagian.append(f"<span class='bintang'>⭐ {float(rr):.2f}</span>")
    if pd.notna(jr):
        bagian.append(f"<span class='ulasan'>{int(jr)} ulasan</span>")
    if not bagian:
        return ""
    return "<div class='kartu-rating'>" + "  ·  ".join(bagian) + "</div>"


def tampilkan_daftar(df, mode="kecocokan"):
    if len(df) == 0:
        st.info("Tidak ada game yang cocok dengan filter ini. Coba ubah filternya.")
        return
    skor_max = 1.0
    if mode == "kecocokan" and "skor" in df.columns:
        m = df["skor"].max()
        skor_max = m if (m and m > 0) else 1.0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        judul = str(row["product_title"])
        plat = row.get("platform")
        label = f"#{i} · {judul}"
        if plat and pd.notna(plat) and plat != "Lainnya":
            label += f"  ·  {plat}"

        with st.expander(label):
            # Badge kategori asli
            kat = row.get("categories")
            if pd.notna(kat):
                st.markdown(
                    f"<span class='badge'>{html.escape(str(kat))}</span>",
                    unsafe_allow_html=True,
                )

            # Rating + jumlah ulasan (ditampilkan di SEMUA tab bila tersedia)
            baris_rating = _baris_rating_html(row)
            if baris_rating:
                st.markdown(baris_rating, unsafe_allow_html=True)

            # Tingkat kecocokan (khusus tab rekomendasi). Label dirender sendiri
            # dengan warna gelap supaya tidak tertimpa tema (label bawaan progress
            # bisa jadi putih dan tak terlihat di kartu putih).
            if mode == "kecocokan" and "skor" in row:
                pct = float(row["skor"]) / skor_max
                st.markdown(
                    "<div class='kartu-teks' style='font-weight:600;margin-bottom:4px'>"
                    "Tingkat kecocokan</div>",
                    unsafe_allow_html=True,
                )
                st.progress(min(1.0, max(0.05, pct)))

            # Deskripsi: judul kecil di atas, isi di bawah (warna gelap eksplisit)
            desc = rapikan(row.get("description"))
            st.markdown("<div class='kartu-label'>Deskripsi</div>", unsafe_allow_html=True)
            if desc:
                st.markdown(
                    f"<div class='kartu-teks'>{html.escape(desc)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='kartu-teks' style='opacity:.65'>"
                    "Tidak ada deskripsi untuk game ini.</div>",
                    unsafe_allow_html=True,
                )

            # Fitur (bila ada)
            fitur = rapikan(row.get("features"))
            if fitur:
                st.markdown("<div class='kartu-label'>Fitur</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='kartu-teks'>{html.escape(fitur)}</div>",
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎮 Rekomendasi Game</h1>
        <p>Temukan game berikutnya yang bakal kamu suka, tanpa ribet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    top_n = st.slider("Berapa banyak rekomendasi?", 5, 20, 10)
    st.markdown("---")
    st.markdown("#### Cara pakai")
    st.markdown(
        "- **Cari yang Mirip** — pilih kategori, cari 1 judul, lihat yang serupa\n"
        "- **Buat Kamu** — pilih kategori, pilih beberapa game favorit\n"
        "- **Lagi Hits** — game terpopuler, disaring kategori & rating\n\n"
        "Klik kartu game mana pun untuk melihat deskripsinya."
    )
    st.markdown("---")
    with st.expander("📊 Detail teknis (untuk reviewer)"):
        st.markdown(
            "Tiga metode dibandingkan (Precision@K & Hit Rate):\n\n"
            "1. **Item-Based Collaborative** (terbaik, ~3x baseline)\n"
            "2. **SVD** (model-based)\n"
            "3. **Baseline Populer**\n\n"
            "App memakai metode pemenang (item-based) untuk rekomendasi personal, "
            "dan content-based untuk produk serupa. Filter kategori/rating dan info "
            "rating hanya lapisan tampilan, tidak mengubah input model "
            "(konsistensi train/serve)."
        )

# ----------------------------------------------------------------------
# Tab utama
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Cari yang Mirip", "✨ Buat Kamu", "🔥 Lagi Hits"])

# --- Tab 1: Content-Based (alur bertahap) ---
with tab1:
    st.markdown("#### Pilih satu game, temukan yang mirip")
    st.caption("Langkahnya: pilih kategori (opsional), cari judul gamenya, lalu rekomendasi muncul.")

    plat1 = st.selectbox("1. Kategori (opsional):", DAFTAR_PLATFORM, key="filter_cb")
    if plat1 == "Semua":
        opsi1 = items.index.tolist()
    else:
        opsi1 = items[items["platform"] == plat1].index.tolist()

    if len(opsi1) == 0:
        st.info("Tidak ada game di kategori ini. Coba kategori lain.")
    else:
        pos = st.selectbox(
            "2. Cari & pilih judul game:",
            options=opsi1,
            format_func=lambda i: items.loc[i, "product_title"],
            index=None,
            placeholder="Ketik nama game untuk mencari...",
            key="cb",
        )
        if pos is None:
            st.info("Cari dan pilih satu game di atas untuk melihat rekomendasinya.")
        else:
            judul = items.loc[pos, "product_title"]
            kat = items.loc[pos, "categories"]
            kat_txt = html.escape(str(kat)) if pd.notna(kat) else ""
            st.markdown(
                f'<div class="selected">🎮 <b>{html.escape(str(judul))}</b>'
                f'<br><span>{kat_txt}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown("##### 3. Game yang mirip (klik untuk detail):")
            tampilkan_daftar(produk_serupa(pos, top_n), mode="kecocokan")

# --- Tab 2: Collaborative (alur bertahap) ---
with tab2:
    st.markdown("#### Pilih game favoritmu, dapatkan rekomendasi")
    st.caption("Langkahnya: pilih kategori (opsional), pilih beberapa game favorit, lalu rekomendasi muncul.")

    plat2 = st.selectbox("1. Kategori (opsional):", DAFTAR_PLATFORM, key="filter_cf")
    if plat2 == "Semua":
        opsi2 = items.index.tolist()
    else:
        opsi2 = items[items["platform"] == plat2].index.tolist()

    # Pastikan game yang sudah dipilih tetap muncul di pilihan, supaya ganti
    # kategori tidak menimbulkan error dan favorit lintas kategori tetap aman.
    terpilih = st.session_state.get("cf", [])
    opsi2 = list(dict.fromkeys(list(terpilih) + opsi2))

    pos_suka = st.multiselect(
        "2. Pilih game favoritmu (boleh lebih dari satu):",
        options=opsi2,
        format_func=lambda i: items.loc[i, "product_title"],
        placeholder="Ketik nama game favoritmu...",
        key="cf",
    )
    if pos_suka:
        st.markdown("##### 3. Mungkin kamu suka (klik untuk detail):")
        tampilkan_daftar(mungkin_suka(pos_suka, top_n), mode="kecocokan")
    else:
        st.info("Pilih minimal satu game favorit untuk mendapatkan rekomendasi.")

# --- Tab 3: Populer (daftar terpopuler, disaring) ---
with tab3:
    st.markdown("#### Game yang lagi hits")
    st.caption(
        "Tab ini menampilkan game terpopuler (paling banyak diulas). "
        "Tidak mencari judul, tetapi bisa disaring berdasarkan kategori dan rating."
    )
    c1, c2 = st.columns(2)
    with c1:
        plat3 = st.selectbox("Kategori:", DAFTAR_PLATFORM, key="filter_pop")
    with c2:
        st.write("")
        hanya_tinggi = st.checkbox("⭐ Hanya rating 4 ke atas", value=False)
    min_r = 4.0 if hanya_tinggi else None
    df_pop = rekomendasi_populer(top_n, min_rating=min_r, platform=plat3)
    tampilkan_daftar(df_pop, mode="populer")

st.markdown("---")
st.caption("Proyek portofolio sistem rekomendasi · dibuat dengan Streamlit")
