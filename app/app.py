import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from datetime import datetime
from streamlit_option_menu import option_menu

# =============================
# 1. KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="TomatAI - Sahabat Petani",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 2. DEFINISI PATH
# =============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, "..")
path_model = os.path.join(ROOT_DIR, "models", "mobilenetv2_tomato.h5")
SAMPLE_DIR = os.path.join(CURRENT_DIR, "sample_images")

# =============================
# 3. STATE MANAGEMENT
# =============================
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(filename, class_name, confidence):
    st.session_state.history.append({
        "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Nama File": filename,
        "Hasil Diagnosa": class_name,
        "Keyakinan": f"{confidence:.2f}%",
        "Status": "✅ Aman" if class_name == "Healthy" else "⚠️ Perlu Tindakan"
    })

# =============================
# 4. CSS TAMPILAN (TEMA PUTIH)
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 95% !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eceef1;
    }

    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.05);
        border-color: #2ea043;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #2ea043;
    }
    .metric-label {
        font-size: 14px;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-box {
        background: #ffffff;
        border-radius: 20px;
        padding: 30px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    
    .btn-wiki {
        display: inline-block;
        padding: 8px 16px;
        color: #2ea043 !important;
        background-color: rgba(46, 160, 67, 0.08);
        border: 1px solid rgba(46, 160, 67, 0.2);
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
    }

    h1, h2, h3 { color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# =============================
# 5. DATABASE PENGETAHUAN
# =============================
CLASS_INFO = {
    "Early Blight": {
        "display_name": "Early Blight (Bercak Kering)",
        "desc": "Penyebabnya jamur *Alternaria solani*. Muncul bercak cokelat konsentris seperti target panahan.",
        "solusi": "Pangkas daun bawah, gunakan fungisida bahan aktif Mancozeb.",
        "status": "Waspada", "color": "#e3b341", "wiki": "https://en.wikipedia.org/wiki/Alternaria_solani"
    },
    "Late Blight": {
        "display_name": "Late Blight (Busuk Daun)",
        "desc": "Sangat berbahaya! Disebabkan *Phytophthora infestans*. Daun tampak seperti tersiram air panas.",
        "solusi": "Cabut tanaman terinfeksi, gunakan fungisida sistemik Dimetomorf.",
        "status": "Kritis", "color": "#d73a49", "wiki": "https://en.wikipedia.org/wiki/Phytophthora_infestans"
    },
    "Healthy": {
        "display_name": "Tanaman Sehat",
        "desc": "Kondisi optimal. Daun hijau segar dan batang kokoh.",
        "solusi": "Pertahankan kebersihan lahan dan nutrisi NPK rutin.",
        "status": "Aman", "color": "#2ea043", "wiki": "https://en.wikipedia.org/wiki/Tomato"
    }
}
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# =============================
# 6. LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    if not os.path.exists(path_model): return None
    return tf.keras.models.load_model(path_model)

model = load_model()

# =============================
# 7. SIDEBAR NAVIGASI (KLIK MENU)
# =============================
with st.sidebar:
    st.markdown("""<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/188/188333.png' width='80'></div>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>TomatAI</h2>", unsafe_allow_html=True)
    
    menu = option_menu(
        menu_title=None,
        options=["Cek Penyakit", "Riwayat Saya", "Tentang Aplikasi"],
        icons=["search", "clock-history", "info-circle"],
        default_index=0,
        styles={
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#2ea043"},
        }
    )
    
    st.divider()
    if model:
        st.success("🟢 Sistem Siap")
    else:
        st.error("🔴 Model Hilang")

# =============================
# LOGIKA HALAMAN
# =============================
if menu == "Cek Penyakit":
    st.markdown("<h1 style='text-align: center;'>🔬 Cek Kesehatan Tanaman</h1>", unsafe_allow_html=True)
    
    # Metrik
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-container"><p class="metric-value">3</p><p class="metric-label">Kategori</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-container"><p class="metric-value">{len(st.session_state.history)}</p><p class="metric-label">Riwayat</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-container"><p class="metric-value" style="font-size:18px">MobileNetV2</p><p class="metric-label">Arsitektur</p></div>', unsafe_allow_html=True)

    st.write("")
    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Foto Daun Tomat", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)
                analyze_btn = st.button("🔎 Analisis Sekarang", type="primary", use_container_width=True)

    with col_right:
        if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
            if model:
                with st.spinner("Menganalisis citra..."):
                    img_array = np.array(image.resize((224, 224))) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    pred = model.predict(img_array)
                    idx = np.argmax(pred)
                    conf = float(np.max(pred) * 100)
                    res = CLASS_NAMES[idx]
                    info = CLASS_INFO[res]
                    
                    add_to_history(uploaded_file.name, res, conf)
                    
                    st.markdown(f"""
                        <div class="result-box" style="border-left: 8px solid {info['color']};">
                            <h3>Hasil: {info['display_name']}</h3>
                            <p>Tingkat Keyakinan: <b>{conf:.2f}%</b></p>
                            <p>Status: <b style="color:{info['color']}">{info['status']}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"**Saran:** {info['solusi']}")
                    st.markdown(f"<a href='{info['wiki']}' class='btn-wiki'>Baca Selengkapnya ↗</a>", unsafe_allow_html=True)
            else:
                st.error("Model tidak tersedia.")
        else:
            st.info("Silakan unggah gambar daun untuk memulai diagnosa.")

elif menu == "Riwayat Saya":
    st.title("📊 Riwayat Diagnosa")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
    else:
        st.warning("Belum ada data.")

elif menu == "Tentang Aplikasi":
    st.title("ℹ️ Tentang TomatAI")
    st.write("Aplikasi ini menggunakan Deep Learning untuk membantu petani mendeteksi penyakit daun tomat secara instan.")
    st.write("**Tim Pengembang:** Kelompok 3")
