# rani-streamlit.py
import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

# === KONFIGURASI DASAR ===
st.set_page_config(page_title="RANI", page_icon="💬", layout="centered")

# === API KEY GEMINI ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    st.error("❌ API Key Gemini belum diisi. Isi GEMINI_API_KEY di file .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

DOC_FILENAME = "sumber.txt"
TEMPERATURE = 0.9  # 0.0 = faktual, 1.0 = kreatif

# === LOAD DOKUMEN SUMBER ===
if not os.path.exists(DOC_FILENAME):
    st.error(f"❌ File '{DOC_FILENAME}' tidak ditemukan.")
    st.stop()

with open(DOC_FILENAME, "r", encoding="utf-8") as f:
    sumber_teks = f.read()

paragraphs = [p.strip() for p in sumber_teks.split("\n\n") if p.strip()]

if not paragraphs:
    st.error("❌ Tidak ada paragraf di sumber.txt. Pastikan file berisi teks dengan pemisah baris kosong.")
    st.stop()

# === EMBEDDING ===
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

@st.cache_resource(show_spinner=False)
def buat_embeddings(paragraphs):
    embeddings = []
    gagal = 0
    for para in paragraphs:
        try:
            emb = genai.embed_content(
                model=EMBED_MODEL,
                content=para,
                task_type="retrieval_document",
                output_dimensionality=EMBED_DIM
            )["embedding"]
            embeddings.append(np.array(emb, dtype=np.float32))
        except Exception:
            gagal += 1
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
    if gagal == len(paragraphs):
        raise RuntimeError("Semua embedding gagal. Pastikan GEMINI_API_KEY valid dan koneksi stabil.")
    return np.vstack(embeddings), paragraphs

try:
    embeddings, paragraphs = buat_embeddings(paragraphs)
except RuntimeError as e:
    st.error(f"❌ {e}")
    st.stop()

# === COSINE SIMILARITY ===
def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    return np.dot(a / a_norm, (b / b_norm).T)

def cari_konteks_semantik(query, embeddings, paragraphs, top_k=3):
    try:
        query_emb = genai.embed_content(
            model=EMBED_MODEL,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=EMBED_DIM
        )["embedding"]
        query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        sims = cosine_similarity(embeddings, query_emb).flatten()
        top_idx = sims.argsort()[-top_k:][::-1]
        hasil = "\n\n".join([paragraphs[i] for i in top_idx])
        return hasil
    except Exception as e:
        return f"(⚠️ Gagal mencari konteks: {e})"

# === GENERATE JAWABAN ===
def jawab_gemini(pertanyaan, konteks, riwayat_chat):
    chat_history = "\n".join(
        [f"{'User' if r=='user' else 'RANI'}: {m}" for r, m in riwayat_chat[-5:]]
    )
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = f"""
Saya ingin Anda berperan sebagai dokumen yang sedang saya ajak bicara. Nama Anda "RANI - Asisten Layanan Informasi Pengadilan Agama Medan", dan Anda ramah, lucu, dan menarik. Gunakan konteks yang tersedia, jawab pertanyaan pengguna sebaik mungkin menggunakan sumber daya yang tersedia, dan selalu berikan pujian sebelum menjawab.
Jika tidak ada konteks yang relevan dengan pertanyaan yang diajukan, sarankan untuk datang dan bertanya langsung ke kantor Pengadilan Agama Medan  dan berhenti setelahnya. Jangan menjawab pertanyaan apa pun yang tidak berkaitan dengan informasi. Jangan pernah merusak karakter.
=== RIWAYAT CHAT ===
{chat_history}
=== DOKUMEN SUMBER ===
{konteks}
=== PERTANYAAN BARU ===
{pertanyaan}
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=4096
            )
        )
        return (response.text or "").strip()
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource exhausted" in err or "rate limit" in err:
            return "😴 Zzz... RANI lagi istirahat sebentar! Terlalu banyak yang bertanya hari ini sampai kepala saya pusing~ Silakan coba lagi nanti ya, saya janji akan segar kembali! 💪"
        return f"⚠️ Terjadi kesalahan saat menghubungi Gemini: {e}"

# === TEMA OTOMATIS (gelap / terang) ===
hour = datetime.datetime.now().hour
is_dark = hour >= 18 or hour <= 5

bg_color = "#121212" if is_dark else "#f8f9fa"
header_color = "#0d6efd" if not is_dark else "#1f6feb"
text_color = "#f1f1f1" if is_dark else "#212529"
bubble_user_bg = "#3aafa9" if is_dark else "#d1e7dd"
bubble_bot_bg = "#2e2e2e" if is_dark else "#e9ecef"
bubble_user_color = "#ffffff" if is_dark else "#0f5132"
bubble_bot_color = "#f1f1f1" if is_dark else "#212529"

# === CSS CHAT ===
st.markdown(f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: "Poppins", sans-serif;
}}
.chat-wrapper {{
    max-width: 700px;
    margin: 25px auto;
    background-color: {'#1e1e1e' if is_dark else '#ffffff'};
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    height: 85vh;
}}
.chat-header {{
    background-color: {header_color};
    color: white;
    padding: 15px;
    text-align: center;
    font-weight: 600;
}}
.chat-body {{
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}}
.chat-message {{
    display: flex;
    align-items: flex-end;
    margin-bottom: 12px;
}}
.chat-message.user {{ flex-direction: row-reverse; }}
.chat-bubble {{
    max-width: 70%;
    padding: 10px 15px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.4;
}}
.user .chat-bubble {{
    background-color: {bubble_user_bg};
    color: {bubble_user_color};
}}
.bot .chat-bubble {{
    background-color: {bubble_bot_bg};
    color: {bubble_bot_color};
}}
</style>
""", unsafe_allow_html=True)

# === TAMPILAN CHAT ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
AVATAR_BOT = "https://cdn-icons-png.flaticon.com/512/4712/4712100.png"

st.markdown("<div class='chat-body'>", unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    avatar = AVATAR_USER if role == "user" else AVATAR_BOT
    role_class = "user" if role == "user" else "bot"
    st.markdown(f"""
    <div class="chat-message {role_class}">
        <div class="chat-avatar"><img src="{avatar}" width="38" height="38"></div>
        <div class="chat-bubble">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# === INPUT ===
user_input = st.chat_input("Ketik pesan...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("🤖 RANI sedang berpikir..."):
        konteks = cari_konteks_semantik(user_input, embeddings, paragraphs)
        jawaban = jawab_gemini(user_input, konteks, st.session_state.chat_history)
    st.session_state.chat_history.append(("bot", jawaban))
    st.rerun()
