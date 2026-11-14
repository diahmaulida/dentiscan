"""DentiScan - skrining awal kondisi gigi & mulut (Streamlit) + riwayat."""
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path("model/best_mobilenetv2_finetuned.keras")
DB_PATH = Path("riwayat.db")
IMG_SIZE = (224, 224)

# Urutan ini HARUS sama dengan train_ds.class_names di notebook (alfabetis).
CLASS_NAMES = ["calculus", "cancer", "caries", "gingivitis", "ulcer"]

INFO = {
    "calculus": ("Karang gigi (calculus)",
                 "Plak yang mengeras. Perlu dibersihkan dokter gigi (scaling)."),
    "cancer": ("Lesi mencurigakan (kemungkinan kanker mulut)",
               "Segera periksakan ke dokter/dokter gigi. Jangan menunda."),
    "caries": ("Gigi berlubang (caries)",
               "Perlu ditambal/dirawat dokter gigi sebelum membesar."),
    "gingivitis": ("Radang gusi (gingivitis)",
                   "Jaga kebersihan mulut, sikat gigi teratur, dan periksa ke dokter gigi."),
    "ulcer": ("Sariawan / ulkus mulut",
              "Sering sembuh sendiri. Jika >2 minggu tidak membaik, periksakan ke dokter."),
}

CONFIDENCE_THRESHOLD = 0.60  # di bawah ini hasil dianggap tidak yakin

st.set_page_config(page_title="DentiScan", page_icon="🦷", layout="centered")


# ---------------------------------------------------------------- Database
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS riwayat (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               waktu TEXT NOT NULL,
               label TEXT NOT NULL,
               confidence REAL NOT NULL,
               probs TEXT NOT NULL,
               foto BLOB
           )"""
    )
    return conn


def save_record(label, conf, probs, img, save_photo):
    foto = None
    if save_photo:
        thumb = ImageOps.exif_transpose(img).convert("RGB")
        thumb.thumbnail((320, 320))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=80)
        foto = buf.getvalue()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO riwayat (waktu, label, confidence, probs, foto) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, conf,
             json.dumps({n: float(p) for n, p in zip(CLASS_NAMES, probs)}), foto),
        )


def load_records():
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, waktu, label, confidence, probs, foto FROM riwayat ORDER BY id DESC"
        ).fetchall()


def delete_record(rec_id=None):
    with get_conn() as conn:
        if rec_id is None:
            conn.execute("DELETE FROM riwayat")
        else:
            conn.execute("DELETE FROM riwayat WHERE id = ?", (rec_id,))


# ------------------------------------------------------------------- Model
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    try:
        return keras.models.load_model(
            MODEL_PATH, custom_objects={"preprocess_input": preprocess_input},
            compile=False)
    except Exception:
        # Lambda layer kadang butuh safe_mode=False. Aman karena model milik sendiri.
        return keras.models.load_model(
            MODEL_PATH, custom_objects={"preprocess_input": preprocess_input},
            compile=False, safe_mode=False)


def predict(model, image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    # JANGAN dibagi 255: preprocess_input MobileNetV2 sudah ada DI DALAM model.
    arr = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    return model.predict(arr, verbose=0)[0]


# ---------------------------------------------------------------------- UI
st.title("🦷 DentiScan")
st.caption("Skrining awal kesehatan gigi & mulut berbasis AI")
st.warning("Hasil ini **bukan diagnosis medis**. Selalu konsultasikan dengan dokter gigi.")

if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan di `{MODEL_PATH}`. Lihat README.md.")
    st.stop()

model = load_model()

tab_scan, tab_history = st.tabs(["🔍 Pindai", "🕘 Riwayat"])

# ----- Tab Pindai
with tab_scan:
    source = st.radio("Sumber foto", ["Kamera", "Unggah file"], horizontal=True)
    file = (st.camera_input("Ambil foto area yang bermasalah")
            if source == "Kamera"
            else st.file_uploader("Pilih foto", type=["jpg", "jpeg", "png", "webp"]))

    if file is not None:
        img = Image.open(file)
        st.image(img, caption="Foto Anda", use_container_width=True)

        with st.spinner("Menganalisis..."):
            probs = predict(model, img)

        order = np.argsort(probs)[::-1]
        top = CLASS_NAMES[order[0]]
        conf = float(probs[order[0]])

        st.subheader("Hasil")
        if conf < CONFIDENCE_THRESHOLD:
            st.info(f"Model kurang yakin (kepercayaan {conf:.0%}). Coba foto ulang dengan "
                    "pencahayaan lebih terang dan fokus jelas, atau periksa ke dokter gigi.")
        else:
            title, advice = INFO[top]
            (st.error if top == "cancer" else st.success)(f"**{title}** — {conf:.0%}")
            st.write(advice)

        st.write("Probabilitas tiap kelas:")
        for i in order:
            st.progress(float(probs[i]), text=f"{INFO[CLASS_NAMES[i]][0]}: {probs[i]:.1%}")

        st.caption("Model hanya mengenali 5 kondisi di atas dan tidak punya kelas "
                   "'sehat'. Gigi/gusi normal tetap bisa diberi label salah.")

        st.divider()
        save_photo = st.checkbox("Simpan juga fotonya (versi kecil)", value=False)
        if st.button("💾 Simpan ke riwayat"):
            save_record(top, conf, probs, img, save_photo)
            st.success("Tersimpan. Lihat di tab Riwayat.")

# ----- Tab Riwayat
with tab_history:
    records = load_records()
    if not records:
        st.info("Belum ada riwayat. Pindai foto lalu tekan 'Simpan ke riwayat'.")
    else:
        st.write(f"Total: **{len(records)}** pemeriksaan")
        for rec_id, waktu, label, conf, probs_json, foto in records:
            with st.container(border=True):
                col_img, col_txt = st.columns([1, 3])
                if foto:
                    col_img.image(foto, use_container_width=True)
                else:
                    col_img.caption("(foto tidak disimpan)")
                col_txt.markdown(f"**{INFO[label][0]}**  \n{waktu} · kepercayaan {conf:.0%}")
                if conf < CONFIDENCE_THRESHOLD:
                    col_txt.caption("⚠️ Kepercayaan rendah saat itu")
                if col_txt.button("🗑️ Hapus", key=f"del_{rec_id}"):
                    delete_record(rec_id)
                    st.rerun()

        st.divider()
        if st.checkbox("Saya yakin ingin menghapus SEMUA riwayat"):
            if st.button("Hapus semua riwayat", type="primary"):
                delete_record()
                st.rerun()