import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from datetime import datetime

# =============================
# 1. KONFIGURASI HALAMAN & TEMA
# =============================
st.set_page_config(
    page_title="TomatAI - Sahabat Petani Tomat",
    page_icon="🍅",
    layout="wide", # Pastikan ini tetap "wide"
    initial_sidebar_state="expanded"
)

# =============================
# 2. STATE MANAGEMENT
# =============================
if 'history' not in st.session_state:
    st.session_state.history = []

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def add_to_history(filename, class_name, confidence):
    """Menyimpan log aktivitas ke session state."""
    st.session_state.history.append({
        "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Nama File": filename,
        "Hasil Diagnosa": class_name,
        "Keyakinan": f"{confidence:.2f}%",
        "Status": "✅ Aman" if class_name == "Healthy" else "⚠️ Perlu Tindakan"
    })

# =============================
# 3. CSS TAMPILAN
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background Gelap */
    .stApp {
        background: linear-gradient(to bottom right, #0d1117, #161b22);
        color: #e6edf3;
    }

    /* --- PERBAIKAN LEBAR HALAMAN DI SINI --- */
    /* Memaksa konten menggunakan lebar maksimal layar */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 98% !important; /* Gunakan 98% lebar layar */
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }

    /* Kartu Metrik */
    .metric-container {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        margin-top: 5px;
    }

    /* Kotak Hasil */
    .result-box {
        background: rgba(33, 38, 45, 0.95);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    
    /* Tombol Link */
    .btn-wiki {
        display: inline-block;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #58a6ff !important;
        background-color: rgba(56, 139, 253, 0.1);
        border: 1px solid rgba(56, 139, 253, 0.4);
        border-radius: 20px;
        text-decoration: none;
        margin-top: 10px;
    }

    /* Progress Bar Hijau */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #238636, #2ea043);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 4. DATABASE PENGETAHUAN
# =============================
CLASS_INFO = {
    "Early Blight": {
        "display_name": "Early Blight (Bercak Kering)",
        "desc": """
        ### 📖 Pengertian & Penyebab
        Ini adalah penyakit **Bercak Kering** atau hawar daun awal. Penyebabnya adalah jamur *Alternaria solani*. Penyakit ini sering muncul saat cuaca mulai lembap atau sering hujan diselingi panas.
        
        ### 🔍 Ciri-Ciri Fisik (Lihat Daunnya):
        1.  **Bercak Bulat:** Ada bercak cokelat atau hitam yang bentuknya bulat-bulat cincin seperti **papan target panahan** atau obat nyamuk bakar.
        2.  **Daun Menguning:** Pinggiran bercak biasanya dikelilingi warna kuning. Lama-lama daun jadi kering kerontang.
        3.  **Posisi:** Biasanya penyakit ini menyerang **daun paling bawah (tua)** dulu, baru menjalar ke atas.
        """,
        "solusi": """
        ### 🛠️ Solusi & Cara Mengobati
        
        **1. Perawatan Lahan (Tanpa Obat):**
        * **Potong Daun:** Segera potong daun yang kena bercak. Buang jauh-jauh atau bakar, jangan ditaruh di bawah pohon.
        * **Kurangi Kelembapan:** Pangkas tunas air atau daun rimbun di bagian bawah supaya angin lancar dan matahari masuk.
        * **Pakai Mulsa:** Gunakan plastik mulsa perak agar spora jamur di tanah tidak muncrat ke daun saat hujan.
        
        **2. Penyemprotan (Obat/Fungisida):**
        * Gunakan fungisida berbahan aktif **Chlorothalonil** (contoh merek: Daconil) atau **Mancozeb**.
        * Semprot **1 minggu sekali**. Pastikan bagian bawah daun juga kena semprot.
        """,
        "status": "Waspada",
        "color": "#e3b341", # Kuning
        "wiki": "https://en.wikipedia.org/wiki/Alternaria_solani"
    },
    "Late Blight": {
        "display_name": "Late Blight (Busuk Daun)",
        "desc": """
        ### 📖 Pengertian & Penyebab
        Ini adalah penyakit **Busuk Daun** atau hawar daun akhir. Penyebabnya adalah jamur air *Phytophthora infestans*. **Hati-hati!** Ini penyakit paling ganas pada tomat. Bisa bikin gagal panen total dalam hitungan hari kalau hujan terus-menerus.
        
        ### 🔍 Ciri-Ciri Fisik (Lihat Daunnya):
        1.  **Bercak Basah:** Luka pada daun terlihat seperti **disiram air panas** (basah/lebam), warnanya hijau kelabu sampai hitam.
        2.  **Ada Bulu Putih:** Kalau pagi hari atau cuaca lembap, coba lihat bagian **bawah daun**. Biasanya ada serbuk/bulu halus warna putih.
        3.  **Menyebar Cepat:** Batang bisa ikut membusuk hitam, dan buah tomat jadi keras berwarna cokelat.
        """,
        "solusi": """
        ### 🛠️ Solusi & Cara Mengobati
        
        **1. Tindakan Darurat (Wajib Cepat):**
        * **Cabut Tanaman:** Kalau sudah parah, cabut tanaman seakarnya. Masukkan kresek, ikat rapat, lalu buang jauh atau bakar. **JANGAN** dijadikan kompos.
        * **Karantina:** Jangan pegang tanaman sehat setelah memegang tanaman sakit sebelum cuci tangan.
        
        **2. Pengaturan Air:**
        * Jangan menyiram tanaman dari atas (disemprot ke daun) saat sore hari. Usahakan daun kering saat malam tiba.
        
        **3. Penyemprotan (Obat/Fungisida):**
        * **Pencegahan:** Pakai fungisida **Tembaga** (Copper) atau **Mancozeb** sebelum hujan turun.
        * **Pengobatan:** Kalau sudah kena, cari obat yang sifatnya *sistemik* (masuk ke jaringan) seperti bahan aktif **Dimethomorph** atau **Cymoxanil**.
        """,
        "status": "Bahaya / Kritis",
        "color": "#d73a49", # Merah
        "wiki": "https://en.wikipedia.org/wiki/Phytophthora_infestans"
    },
    "Healthy": {
        "display_name": "Healthy (Tanaman Sehat)",
        "desc": """
        ### 📖 Kondisi Tanaman
        **Alhamdulillah!** Tanaman tomat Anda terdeteksi dalam kondisi **SEHAT**. Ini berarti perawatan yang Bapak/Ibu lakukan sudah bagus. Nutrisi tercukupi dan tidak ada serangan hama atau jamur yang terlihat.
        
        ### 🔍 Ciri-Ciri Tanaman Sehat:
        1.  **Warna Daun:** Hijau segar merata, tidak ada bercak kuning atau cokelat.
        2.  **Bentuk Daun:** Daun mekar sempurna, tegak, dan tidak layu/loyo.
        3.  **Batang:** Batang kokoh, bersih, dan berbulu halus.
        """,
        "solusi": """
        ### 🛠️ Tips Merawat Agar Tetap Sehat
        
        **1. Jaga Makanan (Pupuk):**
        * Lanjutkan pemupukan berimbang (NPK).
        * Kalau sudah mulai berbunga/berbuah, tambahkan pupuk **Kalsium (Ca)** dan **Kalium (K)** agar bunga tidak rontok dan buah tidak busuk pantat.
        
        **2. Jaga Kebersihan:**
        * Cabut rumput liar di sekitar tanaman karena bisa jadi sarang hama.
        * Tetap pantau tanaman setiap 3 hari sekali, terutama kalau musim hujan.
        """,
        "status": "Aman",
        "color": "#2ea043", # Hijau
        "wiki": "https://en.wikipedia.org/wiki/Tomato"
    }
}
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# =============================
# 5. LOAD MODEL (Cached)
# =============================
# @st.cache_resource
# def load_model():
#     path = "../models/mobilenetv2_tomato.h5" 
#     if not os.path.exists(path):
#         path_backup = "../models/mobilenetv2_tomato.h5"
#         if os.path.exists(path_backup):
#             return tf.keras.models.load_model(path_backup)
#         else:
#             return None
#     return tf.keras.models.load_model(path)

# model = load_model()
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

@st.cache_resource
def load_model():
    # 1. Dapatkan lokasi absolut file app.py saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Bangun path menuju file model
    # Logikanya: Dari folder 'app', MUNDUR satu langkah (..), lalu masuk 'models'
    model_path = os.path.join(current_dir, "..", "models", "mobilenetv2_tomato.h5")
    
    # 3. Cek apakah file benar-benar ada (untuk debugging di logs cloud)
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}")
        return None
        
    return tf.keras.models.load_model(model_path)

# =============================
# 6. SIDEBAR NAVIGASI
# =============================
with st.sidebar:
    # --- Link Gambar Diaktifkan Kembali ---
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=90)
    
    st.markdown("""
    <div style="margin-top: -10px; margin-bottom: 20px;">
        <h2 style="margin:0; font-size: 24px;">TomatAI</h2>
        <p style="color: #8b949e; font-size: 12px;">Asisten Pintar Petani</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Pilih Menu:", 
        ["🚀 Cek Penyakit", "📊 Riwayat Saya", "ℹ️ Tentang Aplikasi"],
        index=0
    )
    
    st.markdown("---")
    
    # Status Indikator
    if model:
        st.markdown("""
        <div style="background: rgba(46, 160, 67, 0.15); border: 1px solid #2ea043; padding: 10px; border-radius: 8px; text-align: center;">
            <span style="color: #2ea043; font-weight: bold;">🟢 Sistem Siap</span>
            <br><span style="font-size: 10px; color: #8b949e;">Otak AI Terhubung</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(215, 58, 73, 0.15); border: 1px solid #d73a49; padding: 10px; border-radius: 8px; text-align: center;">
            <span style="color: #d73a49; font-weight: bold;">🔴 Sistem Error</span>
            <br><span style="font-size: 10px; color: #8b949e;">File Model Hilang</span>
        </div>
        """, unsafe_allow_html=True)

# =============================
# HALAMAN 1: DASHBOARD UTAMA
# =============================
if menu == "🚀 Cek Penyakit":
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🔬 Cek Kesehatan Tanaman Tomat</h1>", unsafe_allow_html=True)

    # 1. KARTU METRIK DASHBOARD
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
        <div class="metric-container">
            <p class="metric-value">{len(CLASS_NAMES)}</p>
            <p class="metric-label">Jenis Penyakit Dikenali</p>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="metric-container">
            <p class="metric-value">{len(st.session_state.history)}</p>
            <p class="metric-label">Foto Dicek Hari Ini</p>
        </div>''', unsafe_allow_html=True)
    with c3:
        # Menangani potensi KeyError dengan .get()
        last_status = "-"
        if st.session_state.history:
            last_item = st.session_state.history[-1]
            last_status = last_item.get('Hasil Diagnosa', last_item.get('Prediksi', '-'))

        st.markdown(f'''
        <div class="metric-container">
            <p class="metric-value" style="font-size: 20px;">{last_status}</p>
            <p class="metric-label">Hasil Terakhir</p>
        </div>''', unsafe_allow_html=True)

    st.write("")
    
    # 2. GALERI REFERENSI (Expandable di Dashboard)
    with st.expander("📚 Buka Kamus Penyakit (Contoh Gambar & Penjelasan)"):
        st.markdown("Lihat contoh gambar di bawah ini untuk membandingkan dengan tanaman Bapak/Ibu:")
        cols = st.columns(len(CLASS_NAMES))
        for idx, name in enumerate(CLASS_NAMES):
            with cols[idx]:
                # Logika cari gambar
                img_path = None
                possible_paths = [os.path.join(SAMPLE_DIR, name), os.path.join(SAMPLE_DIR, name.replace(" ", "_"))]
                for p in possible_paths:
                    if os.path.exists(p):
                        files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.png'))]
                        if files: img_path = os.path.join(p, files[0])
                        break
                
                if img_path:
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(f"*(Gambar {name} Belum Tersedia)*")
                
                st.markdown(f"**{CLASS_INFO[name]['display_name']}**")
                
    st.divider()

    # 3. AREA KERJA
    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.subheader("1. Ambil/Upload Foto Daun")
        st.info("💡 Tips: Pastikan foto fokus pada daun yang sakit dan cahayanya terang.")
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "", 
                type=["jpg", "png", "jpeg"], 
                key=f"up_{st.session_state.uploader_key}"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Foto yang akan dicek", use_container_width=True, channels="RGB")
                analyze_btn = st.button("🔍 Cek Penyakit Sekarang", type="primary", use_container_width=True)

    with col_right:
        if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
            if model:
                # Animasi Loading
                progress_text = "Sedang memeriksa daun..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01) # Simulasi proses
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                # --- PROSES PREDIKSI ---
                img_array = np.array(image.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred = model.predict(img_array)
                idx = np.argmax(pred)
                confidence = float(np.max(pred) * 100)
                class_name = CLASS_NAMES[idx]
                info = CLASS_INFO[class_name]

                time.sleep(0.5)
                my_bar.empty()

                # --- HASIL UTAMA ---
                if confidence < 60.0:
                    st.error("⚠️ **Sistem Ragu-Ragu**")
                    st.markdown("""
                    Sistem kurang yakin dengan foto ini. 
                    * Apakah fotonya buram?
                    * Apakah terlalu gelap?
                    * Atau mungkin ini bukan daun tomat?
                    
                    **Silakan coba foto ulang yang lebih jelas.**
                    """)
                else:
                    # Simpan ke history
                    add_to_history(uploaded_file.name, class_name, confidence)

                    # Tampilan Header Hasil
                    st.markdown(f"""
                    <div class="result-box" style="border-left: 10px solid {info['color']};">
                        <h4 style="margin:0; color: #8b949e;">Hasil Pemeriksaan:</h4>
                        <h1 style="margin-top:5px; color: {info['color']}; font-size: 32px;">{info['display_name']}</h1>
                        <hr style="border-color: #30363d;">
                        <p style="margin:0; font-size: 16px;">
                            Tingkat Keyakinan: <b>{confidence:.2f}%</b> <br>
                            Status: <span style="color: {info['color']}; font-weight: bold;">{info['status']}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detail Informasi (2 Kolom)
                    col_desc, col_sol = st.columns(2, gap="medium")
                    
                    with col_desc:
                        with st.container(border=True):
                            st.markdown(info['desc'])
                            st.markdown(f"<a href='{info.get('wiki', '#')}' target='_blank' class='btn-wiki'>Baca Info di Wikipedia ↗</a>", unsafe_allow_html=True)

                    with col_sol:
                        with st.container(border=True):
                            st.markdown(info['solusi'])
                            st.warning("⚠️ **Penting:** Selalu pakai masker saat menyemprot obat tanaman.")

                    # Visualisasi Statistik (VERSI SEDERHANA)
                    st.markdown("### 📊 Kemungkinan Lainnya")
                    st.markdown("Berikut adalah perkiraan sistem:")
                    probs = pred[0]
                    sorted_idx = np.argsort(probs)[::-1]
                    
                    for i in sorted_idx:
                        score = float(probs[i] * 100)
                        if score > 1.0: # Hanya tampilkan yang di atas 1%
                            col_stat_name, col_stat_bar = st.columns([1, 3])
                            with col_stat_name:
                                # Tampilkan nama penyakitnya
                                st.text(f"{CLASS_INFO[CLASS_NAMES[i]]['display_name']}")
                            with col_stat_bar:
                                st.progress(float(probs[i]))
                                st.caption(f"{score:.2f}%")

        elif not uploaded_file:
            st.info("👈 Silakan upload foto daun di sebelah kiri.")
            st.markdown("""
            <div style="text-align: center; opacity: 0.3; padding-top: 50px;">
                <img src="https://cdn-icons-png.flaticon.com/512/3024/3024310.png" width="150">
                <h3>Belum Ada Foto</h3>
            </div>
            """, unsafe_allow_html=True)

# =============================
# HALAMAN 2: RIWAYAT
# =============================
elif menu == "📊 Riwayat Saya":
    st.title("📊 Catatan Pemeriksaan")
    st.markdown("Ini adalah daftar foto yang sudah Bapak/Ibu cek hari ini:")
    
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # Styling Tabel
        st.dataframe(
            df, 
            use_container_width=True
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Simpan (Excel/CSV)", 
                data=csv, 
                file_name="catatan_penyakit_tomat.csv", 
                mime="text/csv",
                type="primary"
            )
        with col2:
            if st.button("🗑️ Hapus Semua Catatan"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("Belum ada data. Silakan cek penyakit dulu di halaman depan.")

# =============================
# HALAMAN 3: TENTANG
# =============================
elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=120)
    st.title("TomatAI v1.0")
    st.caption("Teknologi Canggih untuk Pertanian Indonesia")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Tujuan Kami")
        st.markdown("""
        Membantu petani tomat mengetahui penyakit tanaman lebih cepat, supaya tidak gagal panen dan hasil kebun melimpah.
        """)
        
    with col2:
        st.subheader("🛠️ Teknologi")
        st.markdown("""
        Aplikasi ini menggunakan Kecerdasan Buatan (AI) yang sudah belajar dari ribuan foto daun tomat untuk mengenali penyakit.
        """)

    st.divider()
    st.subheader("👥 Dibuat Oleh (Kelompok 3)")
    
    team_cols = st.columns(4)
    members = ["Achmad Karis Wibowo", "Albert Cendra Hermawan", "Yosia Marpaung", "Dhimas Muhammad Fattah Arrumy"]
    
    for i, member in enumerate(members):
        with team_cols[i]:
            st.markdown(f"""
            <div style="background: #21262d; padding: 10px; border-radius: 8px; text-align: center;">
                <p style="font-weight: bold; margin:0;">{member}</p>
                <p style="font-size: 12px; color: #8b949e;">Tim Pengembang</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><p style='text-align: center; color: #8b949e; font-size: 12px;'>© 2026 TomatAI Project.</p>", unsafe_allow_html=True)
