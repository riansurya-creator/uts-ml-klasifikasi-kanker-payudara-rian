import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sistem Prediksi Kanker Payudara - Random Forest",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model_rf_breast_cancer.pkl")

model = load_model()

st.title("🩺 Aplikasi Klasifikasi Diagnosis Kanker Payudara (WDBC)")
st.markdown("""
**Implementasi Algoritma Random Forest Classifier**  
*Dikembangkan oleh: **Rian Surya** (NIM: 301240058 | Kelas: 5C) — UTS Pembelajaran Mesin*
""")
st.divider()

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📋 Parameter Morfologi Inti Sel (Fine Needle Aspirate)")
    st.caption("Masukkan nilai pengukuran morfologi jaringan atau gunakan tombol preset sampel uji.")

    preset = st.radio(
        "Pilih Preset Contoh Data Klinis:",
        ["Manual / Default", "Contoh Pasien Jinak (Benign)", "Contoh Pasien Ganas (Malignant)"],
        horizontal=True
    )

    if preset == "Contoh Pasien Jinak (Benign)":
        defaults = [11.42, 14.96, 73.58, 403.5, 0.092, 0.075, 0.032, 0.018, 0.172, 0.061]
    elif preset == "Contoh Pasien Ganas (Malignant)":
        defaults = [20.57, 21.77, 136.50, 1326.0, 0.118, 0.195, 0.241, 0.128, 0.215, 0.068]
    else:
        defaults = [14.12, 19.28, 91.96, 654.8, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062]

    c1, c2 = st.columns(2)
    with c1:
        mean_radius = st.number_input("Mean Radius (mm)", 5.0, 35.0, float(defaults[0]), step=0.1)
        mean_texture = st.number_input("Mean Texture", 5.0, 45.0, float(defaults[1]), step=0.1)
        mean_perimeter = st.number_input("Mean Perimeter (mm)", 30.0, 220.0, float(defaults[2]), step=0.5)
        mean_area = st.number_input("Mean Area (mm²)", 100.0, 2600.0, float(defaults[3]), step=5.0)
        mean_smoothness = st.number_input("Mean Smoothness", 0.04, 0.20, float(defaults[4]), step=0.005, format="%.4f")
    with c2:
        mean_compactness = st.number_input("Mean Compactness", 0.01, 0.40, float(defaults[5]), step=0.005, format="%.4f")
        mean_concavity = st.number_input("Mean Concavity", 0.00, 0.50, float(defaults[6]), step=0.005, format="%.4f")
        mean_concave_pts = st.number_input("Mean Concave Points", 0.00, 0.25, float(defaults[7]), step=0.005, format="%.4f")
        mean_symmetry = st.number_input("Mean Symmetry", 0.10, 0.35, float(defaults[8]), step=0.005, format="%.4f")
        mean_fractal = st.number_input("Mean Fractal Dimension", 0.04, 0.12, float(defaults[9]), step=0.002, format="%.4f")

with col2:
    st.subheader("🔍 Hasil Analisis & Prediksi Model")
    features = np.array([[
        mean_radius, mean_texture, mean_perimeter, mean_area,
        mean_smoothness, mean_compactness, mean_concavity,
        mean_concave_pts, mean_symmetry, mean_fractal
    ]])

    if st.button("Jalankan Prediksi Diagnosis", type="primary", use_container_width=True):
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        if pred == 1:
            st.error(f"### ⚠️ Terdeteksi Risiko Tinggi: GANAS (Malignant)")
            st.metric("Probabilitas Kanker Ganas", f"{proba[1]*100:.2f}%", delta="Perlu Pemeriksaan Lanjut", delta_color="inverse")
        else:
            st.success(f"### ✅ Terdeteksi Risiko Rendah: JINAK (Benign)")
            st.metric("Probabilitas Tumor Jinak", f"{proba[0]*100:.2f}%", delta="Karakteristik Normal")

        prob_df = pd.DataFrame({
            "Kategori Diagnosis": ["Jinak (Benign)", "Ganas (Malignant)"],
            "Probabilitas (%)": [round(proba[0]*100, 2), round(proba[1]*100, 2)]
        })
        st.bar_chart(prob_df.set_index("Kategori Diagnosis"))
        st.info("Catatan Akademik: Aplikasi ini merupakan prototipe evaluasi model Random Forest (Akurasi Uji: 93.86%, ROC-AUC: 0.985) untuk keperluan tugas UTS Pembelajaran Mesin.")
