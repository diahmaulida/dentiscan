import io
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# PENGATURAN
#   "lokal" -> riwayat disimpan di file riwayat.db (tahan restart, untuk dipakai sendiri)
#   "sesi"  -> riwayat hanya ada selama tab browser terbuka (aman untuk website publik)
# ------------------------------------------------------------------
MODE = "lokal"
DB_PATH = Path("riwayat.db")

st.set_page_config(page_title="DentiScan", page_icon="🦷")

LABEL = {
    "caries": "Gigi berlubang (karies)",
    "ulcer": "Sariawan / ulkus",
    "tooth discoloration": "Perubahan warna gigi",
    "gingivitis": "Radang gusi",
}


def terjemah(nama):
    return LABEL.get(nama.lower(), nama)


# Informasi edukasi umum per kondisi (bukan pengganti pemeriksaan dokter gigi)
INFO = {
    "caries": {
        "penyebab": [
            "Bakteri plak mengubah gula menjadi asam yang melarutkan email gigi.",
            "Sering makan/minum manis atau lengket, terutama di antara waktu makan.",
            "Menyikat gigi kurang rutin atau tidak memakai pasta gigi berfluoride.",
            "Mulut kering (kurang air liur) dan sisa makanan terselip di sela gigi.",
        ],
        "solusi": [
            "Sikat gigi 2x sehari dengan pasta gigi berfluoride, bersihkan sela gigi dengan benang gigi.",
            "Kurangi makanan dan minuman manis, terutama sebelum tidur.",
            "Lubang gigi tidak bisa sembuh sendiri: perlu ditambal atau dirawat dokter gigi.",
            "Kontrol ke dokter gigi setiap 6 bulan.",
        ],
        "dokter": "Segera periksa jika gigi nyeri berdenyut, bengkak, atau ada nanah.",
    },
    "ulcer": {
        "penyebab": [
            "Luka akibat tergigit atau tergesek kawat gigi/gigi tajam.",
            "Stres, kurang tidur, atau perubahan hormon.",
            "Kekurangan zat besi, vitamin B12, atau asam folat.",
            "Iritasi bahan pasta gigi tertentu, makanan pedas/asam, atau infeksi virus.",
        ],
        "solusi": [
            "Hindari makanan pedas, asam, dan keras selama sariawan.",
            "Jaga kebersihan mulut, berkumur air garam hangat atau obat kumur antiseptik.",
            "Gel/obat oles pelindung sariawan dari apotek dapat membantu meredakan nyeri.",
            "Perbanyak air putih, sayur, dan buah. Sariawan biasa umumnya sembuh dalam 1-2 minggu.",
        ],
        "dokter": "Periksakan jika luka tidak sembuh lebih dari 2-3 minggu, makin membesar, "
                  "sering kambuh, atau disertai demam. Luka mulut yang menetap perlu "
                  "dinilai dokter untuk menyingkirkan penyebab serius.",
    },
    "tooth discoloration": {
        "penyebab": [
            "Noda dari kopi, teh, minuman bersoda, dan rokok.",
            "Plak atau karang gigi yang menguning.",
            "Gigi berlubang, trauma pada gigi, atau gigi yang pernah dirawat saluran akar.",
            "Penggunaan obat tertentu saat gigi tumbuh, kelebihan fluoride, dan penuaan.",
        ],
        "solusi": [
            "Bersihkan karang gigi (scaling) di dokter gigi secara berkala.",
            "Sikat gigi teratur, kumur setelah minum kopi/teh, kurangi rokok.",
            "Pemutihan gigi (bleaching) sebaiknya dilakukan atau diawasi dokter gigi.",
            "Untuk noda yang tidak hilang, dokter dapat menyarankan veneer atau tambalan estetik.",
        ],
        "dokter": "Periksa jika hanya satu gigi yang menggelap, karena bisa tanda gigi "
                  "mati atau lubang di dalam.",
    },
    "gingivitis": {
        "penyebab": [
            "Plak dan karang gigi menumpuk di garis gusi.",
            "Menyikat gigi kurang bersih atau tidak membersihkan sela gigi.",
            "Merokok, diabetes, dan perubahan hormon (misalnya kehamilan).",
            "Obat tertentu, mulut kering, atau kekurangan vitamin C.",
        ],
        "solusi": [
            "Sikat gigi 2x sehari dengan sikat lembut hingga ke garis gusi, tanpa menekan keras.",
            "Gunakan benang gigi dan obat kumur antiseptik.",
            "Bersihkan karang gigi di dokter gigi.",
            "Dengan kebersihan yang baik, radang gusi ringan biasanya membaik dalam 1-2 minggu.",
        ],
        "dokter": "Periksa jika gusi terus berdarah, bengkak, bau mulut menetap, atau gigi "
                  "terasa goyang, karena bisa berkembang menjadi radang jaringan penyangga gigi.",
    },
}


def tampilkan_info(kunci_list, dalam_expander=False):
    """Tampilkan penyebab & solusi untuk tiap kondisi unik yang terdeteksi."""
    for k in dict.fromkeys(kunci_list):
        info = INFO.get(str(k).lower())
        if not info:
            continue
        judul = f"ℹ️ {terjemah(k)}: penyebab & solusi"
        # Streamlit tidak mengizinkan expander di dalam expander
        wadah = st.container() if dalam_expander else st.expander(judul)
        with wadah:
            if dalam_expander:
                st.markdown(f"**{judul}**")
            st.markdown("**Kemungkinan penyebab**")
            for p in info["penyebab"]:
                st.markdown(f"- {p}")
            st.markdown("**Yang bisa dilakukan**")
            for s in info["solusi"]:
                st.markdown(f"- {s}")
            st.markdown(f"**Kapan ke dokter gigi:** {info['dokter']}")
            st.caption("Informasi umum, bukan pengganti pemeriksaan dokter gigi.")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


# ------------------------------------------------------------------
# PENYIMPANAN RIWAYAT
# ------------------------------------------------------------------
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS riwayat(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT, nama_file TEXT, conf REAL,
            temuan TEXT, gambar BLOB)"""
    )
    return con


def simpan(nama_file, conf, temuan, gambar_bytes):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if MODE == "lokal":
        with closing(_db()) as con, con:
            con.execute(
                "INSERT INTO riwayat(waktu, nama_file, conf, temuan, gambar) "
                "VALUES (?,?,?,?,?)",
                (waktu, nama_file, conf, json.dumps(temuan), gambar_bytes),
            )
    else:
        st.session_state.setdefault("riwayat", []).append(
            dict(id=uuid.uuid4().hex, waktu=waktu, nama_file=nama_file,
                 conf=conf, temuan=json.dumps(temuan), gambar=gambar_bytes)
        )


def daftar():
    """Semua riwayat, terbaru di atas."""
    if MODE == "lokal":
        with closing(_db()) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM riwayat ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]
    return list(reversed(st.session_state.get("riwayat", [])))


def hapus(rid):
    if MODE == "lokal":
        with closing(_db()) as con, con:
            con.execute("DELETE FROM riwayat WHERE id = ?", (rid,))
    else:
        st.session_state["riwayat"] = [
            r for r in st.session_state.get("riwayat", []) if r["id"] != rid
        ]


def hapus_semua():
    if MODE == "lokal":
        with closing(_db()) as con, con:
            con.execute("DELETE FROM riwayat")
    else:
        st.session_state["riwayat"] = []


# ------------------------------------------------------------------
# TAMPILAN
# ------------------------------------------------------------------
st.title("🦷 DentiScan")
st.caption("Skrining awal kondisi gigi dan mulut berbasis AI")

model = load_model()

with st.sidebar:
    st.header("Pengaturan")
    conf = st.slider("Ambang kepercayaan", 0.10, 0.90, 0.25, 0.05)
    st.caption("Kondisi yang dikenali model:")
    for n in model.names.values():
        st.write("•", terjemah(n))
    st.caption(
        "DentiScan tidak dapat mendeteksi kanker mulut, karang gigi, "
        "atau penyakit lain di luar daftar ini."
    )
    st.divider()
    if MODE == "lokal":
        st.caption("📁 Riwayat disimpan di file riwayat.db pada komputer ini.")
    else:
        st.caption("🕘 Riwayat hanya tersimpan selama tab ini terbuka.")

tab_deteksi, tab_riwayat = st.tabs(["🔍 Deteksi", "🕘 Riwayat"])

# ---------------- TAB DETEKSI ----------------
with tab_deteksi:
    file = st.file_uploader("Unggah foto gigi/mulut", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file).convert("RGB")
        res = model(img, conf=conf)[0]
        hasil_img = Image.fromarray(res.plot()[:, :, ::-1])
        st.image(hasil_img, caption="Hasil deteksi", use_container_width=True)

        temuan = [
            {
                "kunci": model.names[int(b.cls)],
                "kondisi": terjemah(model.names[int(b.cls)]),
                "keyakinan": round(float(b.conf), 2),
            }
            for b in res.boxes
        ]

        if not temuan:
            st.info(
                "Tidak ada temuan pada foto ini. Ini TIDAK berarti mulut Anda sehat, "
                "karena DentiScan hanya mengenali 4 kondisi."
            )
        else:
            st.subheader("Temuan")
            for t in temuan:
                st.write(f"- **{t['kondisi']}**: {t['keyakinan']:.0%}")
            st.subheader("Penyebab & solusi")
            tampilkan_info([t["kunci"] for t in temuan])

        st.warning(
            "DentiScan hanya alat skrining awal, bukan diagnosis. "
            "Periksakan ke dokter gigi untuk kepastian."
        )

        st.divider()
        sertakan_foto = st.checkbox(
            "Sertakan foto hasil deteksi di riwayat",
            value=False,
            help="Foto mulut adalah data kesehatan. Biarkan mati jika tidak perlu.",
        )
        if st.button("💾 Simpan ke riwayat"):
            gambar_bytes = None
            if sertakan_foto:
                buf = io.BytesIO()
                hasil_img.save(buf, format="JPEG", quality=85)
                gambar_bytes = buf.getvalue()
            simpan(file.name, conf, temuan, gambar_bytes)
            st.success("Tersimpan. Lihat di tab Riwayat.")

# ---------------- TAB RIWAYAT ----------------
with tab_riwayat:
    data = daftar()

    if not data:
        st.info("Belum ada riwayat. Lakukan deteksi lalu klik 'Simpan ke riwayat'.")
    else:
        # Tabel ringkasan + unduh CSV
        ringkas = []
        for r in data:
            t = json.loads(r["temuan"])
            ringkas.append(
                {
                    "Waktu": r["waktu"],
                    "File": r["nama_file"],
                    "Jumlah temuan": len(t),
                    "Kondisi": ", ".join(sorted({x["kondisi"] for x in t})) or "-",
                    "Ambang": r["conf"],
                }
            )
        df = pd.DataFrame(ringkas)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Unduh riwayat (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "riwayat_dentiscan.csv",
            "text/csv",
        )

        st.subheader("Detail")
        for r in data:
            t = json.loads(r["temuan"])
            with st.expander(f"{r['waktu']} — {r['nama_file']} — {len(t)} temuan"):
                if t:
                    st.table(pd.DataFrame(t)[["kondisi", "keyakinan"]])
                    tampilkan_info([x.get("kunci", "") for x in t], dalam_expander=True)
                else:
                    st.write("Tidak ada temuan.")
                if r["gambar"]:
                    st.image(r["gambar"], use_container_width=True)
                if st.button("🗑️ Hapus catatan ini", key=f"del_{r['id']}"):
                    hapus(r["id"])
                    st.rerun()

        st.divider()
        yakin = st.checkbox("Saya yakin ingin menghapus SEMUA riwayat")
        if st.button("Hapus semua riwayat", disabled=not yakin):
            hapus_semua()
            st.rerun()