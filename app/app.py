import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from datetime import datetime

# =========================================================
# 1. KONFIGURASI HALAMAN (Nuansa Bersih & Profesional)
# =========================================================
st.set_page_config(
    page_title="TomatAI - Solusi Pintar Budidaya Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. PATHING & DATABASE (Tetap pada Struktur Folder Anda)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")
path_model = os.path.join(ROOT_DIR, "models", "mobilenetv2_tomato.h5")
SAMPLE_DIR = os.path.join(CURRENT_DIR, "sample_images")

# Informasi Penyakit dengan Bahasa yang Lebih Manusiawi
CLASS_INFO = {
    "Early Blight": {
        "display_name": "Bercak Kering (Early Blight)",
        "desc": "Penyakit ini disebabkan oleh jamur *Alternaria solani*. Biasanya menyerang daun tua terlebih dahulu dengan ciri khas bercak cokelat konsentris yang menyerupai papan target panahan.",
        "solusi": "Potong dan musnahkan daun yang terinfeksi. Gunakan fungisida berbahan aktif mankozeb atau klorotalonil sesuai dosis anjuran.",
        "status": "Waspada",
        "color": "#F39C12",
        "wiki": "https://id.wikipedia.org/wiki/Alternaria_solani"
    },
    "Late Blight": {
        "display_name": "Busuk Daun (Late Blight)",
        "desc": "Ini kategori serius! Disebabkan oleh *Phytophthora infestans*. Tandanya adalah bercak basah hijau kelabu yang cepat sekali meluas hingga membuat tanaman tampak seperti tersiram air panas.",
        "solusi": "Segera isolasi tanaman yang sakit. Gunakan fungisida sistemik berbahan aktif dimetomorf atau simoksanil untuk menekan penyebaran.",
        "status": "Bahaya / Kritis",
        "color": "#E74C3C",
        "wiki": "https://id.wikipedia.org/wiki/Phytophthora_infestans"
    },
    "Healthy": {
        "display_name": "Tanaman Sehat (Normal)",
        "desc": "Luar biasa! Tanaman Anda menunjukkan tanda-tanda pertumbuhan yang optimal. Daun berwarna hijau segar, tekstur kaku, dan tidak ada indikasi serangan patogen.",
        "solusi": "Pertahankan pola pemupukan rutin dan pastikan drainase lahan tetap terjaga agar kelembapan tidak memicu jamur.",
        "status": "Aman",
        "color": "#27AE60",
        "wiki": "https://id.wikipedia.org/wiki/Tomat"
    }
}
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# =========================================================
# 3. CSS CUSTOM (Fokus pada Background Putih & Tipografi)
# =========================================================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #FFFFFF !important;
        color: #2C3E50;
    }
    
    /* Sidebar Putih dengan Border Tipis */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }

    /* Card Metrik yang Elegan */
    .card-metric {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: 0.3s;
    }
    .card-metric:hover {
        border-color: #27AE60;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 30px; font-weight: 700; color: #2C3E50; margin: 0; }
    .metric-lbl { font-size: 14px; color: #7F8C8D; margin-top: 5px; }

    /* Hasil Diagnosa Box */
    .result-container {
        background: #F8F9FA;
        border-radius: 16px;
        padding: 30px;
        border: 1px solid #E9ECEF;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. LOGIC & STATE
# =========================================================
if 'history' not in st.session_state: st.session_state.history = []
if 'menu_active' not in st.session_state: st.session_state.menu_active = "🏠 Beranda"

@st.cache_resource
def load_tomato_model():
    if os.path.exists(path_model):
        return tf.keras.models.load_model(path_model)
    return None

model = load_tomato_model()

# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🍅 TomatAI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#7F8C8D;'>Versi Mahasiswa 1.0</p>", unsafe_allow_html=True)
    st.write("---")
    
    nav_options = ["🏠 Beranda", "📊 Riwayat Cek", "👨‍💻 Tentang Kami"]
    for opt in nav_options:
        if st.button(opt, use_container_width=True, type="primary" if st.session_state.menu_active == opt else "secondary"):
            st.session_state.menu_active = opt
            st.rerun()

# =========================================================
# 6. HALAMAN: BERANDA
# =========================================================
if st.session_state.menu_active == "🏠 Beranda":
    st.markdown("# Diagnosa Kesehatan Tanaman")
    st.markdown("Unggah foto daun tomat Anda untuk mendapatkan hasil analisis instan berbasis AI.")
    
    # Dashboard Mini
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="card-metric"><p class="metric-val">3</p><p class="metric-lbl">Kategori Terdeteksi</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card-metric"><p class="metric-val">{len(st.session_state.history)}</p><p class="metric-lbl">Analisis Hari Ini</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="card-metric"><p class="metric-val">90%+</p><p class="metric-lbl">Akurasi Model</p></div>', unsafe_allow_html=True)

    st.write("---")

    col_up, col_res = st.columns([1, 1.2], gap="large")

    with col_up:
        st.subheader("📸 Ambil Gambar")
        uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Pratinjau Gambar", use_container_width=True)
            run_btn = st.button("Mulai Analisis Sekarang", type="primary", use_container_width=True)

    with col_res:
        if uploaded_file and 'run_btn' in locals() and run_btn:
            if model:
                with st.spinner("Menganalisis tekstur daun..."):
                    # Simulasi & Prediksi
                    img_resized = np.array(img.resize((224, 224))) / 255.0
                    prediction = model.predict(np.expand_dims(img_resized, axis=0))
                    idx = np.argmax(prediction)
                    conf = float(np.max(prediction) * 100)
                    res_info = CLASS_INFO[CLASS_NAMES[idx]]
                    
                    time.sleep(1) # Biar ada feel "mikir"
                    
                    if conf > 65:
                        # Tampilan Hasil
                        st.markdown(f"""
                        <div class="result-container" style="border-top: 5px solid {res_info['color']};">
                            <span style="color:{res_info['color']}; font-weight:bold;">STATUS: {res_info['status']}</span>
                            <h2 style="margin: 10px 0;">{res_info['display_name']}</h2>
                            <p style="font-size:18px;">Tingkat Keyakinan: <b>{conf:.2f}%</b></p>
                            <hr>
                            <h4>Analisis Singkat:</h4>
                            <p>{res_info['desc']}</p>
                            <h4>Rekomendasi Tindakan:</h4>
                            <p>{res_info['solusi']}</p>
                            <br>
                            <a href="{res_info['wiki']}" target="_blank" style="color:#27AE60; text-decoration:none; font-weight:bold;">Pelajari Selengkapnya di Wiki ↗</a>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Simpan History
                        st.session_state.history.append({
                            "Waktu": datetime.now().strftime("%H:%M"),
                            "Hasil": res_info['display_name'],
                            "Akurasi": f"{conf:.1f}%"
                        })
                    else:
                        st.warning("⚠️ Hasil kurang meyakinkan. Coba ambil foto ulang dengan cahaya yang lebih baik dan posisi daun yang lebih fokus.")
            else:
                st.error("Model AI tidak ditemukan. Hubungi tim pengembang.")
        else:
            st.info("Menunggu gambar diunggah untuk memulai diagnosa.")

# =========================================================
# 7. HALAMAN: RIWAYAT & TENTANG (Diringkas untuk contoh)
# =========================================================
elif st.session_state.menu_active == "📊 Riwayat Cek":
    st.title("Riwayat Pemeriksaan")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("Belum ada riwayat pengecekan untuk sesi ini.")

elif st.session_state.menu_active == "👨‍💻 Tentang Kami":
    st.title("Tentang Proyek")
    st.write("Aplikasi ini dikembangkan oleh Kelompok 3 sebagai bagian dari tugas mata kuliah Artificial Intelligence. Fokus utama kami adalah membantu digitalisasi sektor pertanian lokal.")
    
    st.subheader("Tim Pengembang")
    st.write("- Achmad Karis Wibowo\n- Albert Cendra Hermawan\n- Yosia Marpaung\n- Dhimas Muhammad Fattah Arrumy")
