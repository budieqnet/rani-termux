import streamlit as st
import google.generativeai as genai
import numpy as np
import os
import time
import datetime
import json
from dotenv import load_dotenv
from streamlit.components.v1 import html

load_dotenv()

# ================== KONFIGURASI ==================
st.set_page_config(
    page_title="RANI - PA Medan",
    page_icon="⚖️",
    layout="centered"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_FILENAME = os.path.join(SCRIPT_DIR, "sumber.txt")
TEMPERATURE = 0.9

if not GEMINI_API_KEY:
    st.error("❌ API Key Gemini belum diisi. Isi GEMINI_API_KEY di file .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ================== STATE ==================
for k, v in {
    "chat_history": [],
    "is_typing": False,
    "gesture": "idle",
    "last_message_time": 0,
    "voice_gender": "female",
    "voice_listening": False,
    "processing": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================== LOAD DOKUMEN ==================
if not os.path.exists(DOC_FILENAME):
    st.error(f"❌ File sumber.txt tidak ditemukan di: {DOC_FILENAME}")
    st.stop()

with open(DOC_FILENAME, "r", encoding="utf-8") as f:
    paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

if not paragraphs:
    st.error("❌ Tidak ada paragraf di sumber.txt. Pastikan file berisi teks dengan pemisah baris kosong.")
    st.stop()

# ================== EMBEDDING === (sama seperti rani-streamlit.py)
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

@st.cache_resource(show_spinner=False)
def buat_embedding(paras):
    embeddings = []
    gagal = 0
    for p in paras:
        try:
            e = genai.embed_content(
                model=EMBED_MODEL,
                content=p,
                task_type="retrieval_document",
                output_dimensionality=EMBED_DIM
            )["embedding"]
            embeddings.append(np.array(e, dtype=np.float32))
        except Exception:
            gagal += 1
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
    if gagal == len(paras):
        raise RuntimeError("Semua embedding gagal. Pastikan GEMINI_API_KEY valid dan koneksi stabil.")
    return np.vstack(embeddings), paras

try:
    embeddings, paragraphs = buat_embedding(paragraphs)
except RuntimeError as e:
    st.error(f"❌ {e}")
    st.stop()

def cosine_sim(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)
    return np.dot(a / a_norm, (b / b_norm).T)

def cari_konteks(q, k=3):
    try:
        q_emb = genai.embed_content(
            model=EMBED_MODEL,
            content=q,
            task_type="retrieval_query",
            output_dimensionality=EMBED_DIM
        )["embedding"]
        q_emb = np.array(q_emb, dtype=np.float32).reshape(1, -1)
        sims = cosine_sim(embeddings, q_emb).flatten()
        idx = sims.argsort()[-k:][::-1]
        return "\n\n".join(paragraphs[i] for i in idx)
    except Exception:
        return ""

# ================== GEMINI ==================
def jawab_gemini(tanya, konteks, history):
    riwayat = "\n".join(
        [f"{'User' if r=='user' else 'RANI'}: {m}" for r, m in history[-5:]]
    )

    prompt = f"""
Saya ingin Anda berperan sebagai dokumen yang sedang saya ajak bicara. Nama Anda "RANI - Asisten Layanan Informasi Pengadilan Agama Medan" dan Anda ramah dan menarik dan gunakan karakter suara yang ramah dan menarik juga. Gunakan konteks yang tersedia, jawab pertanyaan pengguna sebaik mungkin menggunakan sumber daya yang tersedia, berikan jawaban yang lengkap, jelas dan jangan memotong jawaban di tengah kalimat.
Jika tidak ada konteks yang relevan dengan pertanyaan yang diajukan, sarankan untuk datang dan bertanya langsung ke kantor Pengadilan Agama Medan  dan berhenti setelahnya. Jangan menjawab pertanyaan apa pun yang tidak berkaitan dengan informasi. Jangan pernah merusak karakter.

PENTING: Berikan jawaban yang lengkap dan jelas. Jangan memotong jawaban di tengah kalimat.

=== RIWAYAT ===
{riwayat}
=== KONTEKS ===
{konteks}
=== PERTANYAAN ===
{tanya}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        res = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=8192
            )
        )
        return (res.text or "").strip()
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource exhausted" in err or "rate limit" in err:
            return "😴 Zzz... RANI lagi istirahat sebentar! Terlalu banyak yang bertanya hari ini sampai kepala saya pusing~ Silakan coba lagi nanti ya, saya janji akan segar kembali! 💪"
        return f"⚠️ Terjadi kesalahan saat menghubungi Gemini: {e}"

# ================== FILTER SPAM ==================
def filter_spam(text):
    if len(text) < 3:
        return False
    blacklist = ["http", "@@@", "!!!", "spam"]
    return not any(b in text.lower() for b in blacklist)

# ================== TTS - PERBAIKAN UTAMA ==================
def rani_bicara(teks):
    # 🔹 Bersihkan karakter markdown dan special chars
    bersih = teks.replace("*", "").replace("#", "").replace("\n", " ")
    bersih = " ".join(bersih.split())  # Rapikan spasi
    
    # 🔹 Batasi panjang untuk TTS (max ~500 karakter per chunk)
    max_length = 500
    chunks = []
    
    if len(bersih) <= max_length:
        chunks = [bersih]
    else:
        # Potong di titik atau koma terdekat
        words = bersih.split()
        current = []
        current_len = 0
        
        for word in words:
            if current_len + len(word) + 1 > max_length:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        
        if current:
            chunks.append(" ".join(current))
    
    # 🔹 Encode text dengan JSON untuk mencegah masalah quote/escape
    chunks_json = json.dumps(chunks)
    
    html(f"""
    <script>
    (function() {{
        const synth = window.speechSynthesis;
        synth.cancel();
        
        const chunks = {chunks_json};
        let currentIndex = 0;
        
        function speakNext() {{
            if (currentIndex >= chunks.length) return;
            
            const utterance = new SpeechSynthesisUtterance(chunks[currentIndex]);
            utterance.lang = 'id-ID';
            utterance.rate = 0.95;
            utterance.pitch = 1.1;
            
            utterance.onend = function() {{
                currentIndex++;
                if (currentIndex < chunks.length) {{
                    setTimeout(speakNext, 200); // Jeda 200ms antar chunk
                }}
            }};
            
            utterance.onerror = function(e) {{
                console.error('Speech error:', e);
                currentIndex++;
                if (currentIndex < chunks.length) {{
                    setTimeout(speakNext, 200);
                }}
            }};
            
            synth.speak(utterance);
        }}
        
        speakNext();
    }})();
    </script>
    """, height=0)


# ================== AVATAR & GESTURE ==================
GESTURE = {
    "idle":  "https://assets6.lottiefiles.com/packages/lf20_tno6cg2w.json",
    "think": "https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json",
    "speak": "https://assets9.lottiefiles.com/packages/lf20_kyu7xb1v.json",
    "smile": "https://assets3.lottiefiles.com/packages/lf20_xlmz9xwm.json"
}

html(f"""
<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>

<lottie-player
 id="rani-avatar"
 src="{GESTURE[st.session_state.gesture]}"
 background="transparent"
 speed="1"
 style="
   position: fixed;
   bottom: 40px;
   right: 20px;
   width: 360px;
   height: 360px;
   opacity: 0.6;
   z-index: 999;
   pointer-events: none;
 "
 loop
 autoplay>
</lottie-player>

<script>
document.addEventListener("scroll", function() {{
  const a = document.getElementById("rani-avatar");
  if (!a) return;
  a.style.transform =
    "translateY(" + (-window.scrollY * 0.03) + "px)";
}});
</script>
""", height=0)


# ================== CSS ==================
st.markdown("""
<style>
.stApp {
    background: transparent;
}
.chat-wrapper {
    z-index: 5;
    position: relative;
}
.chat-message {
    animation: fade .3s;
}
@keyframes fade {
    from {opacity:0; transform:translateY(10px)}
    to {opacity:1}
}
/* Custom title size */
h1 {
    text-align: center !important;
    font-size: 1.5rem !important;
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ================== CHAT ==================
st.title("⚖️ RANI – Layanan Informasi Pengadilan Agama Medan")

for r, m in st.session_state.chat_history:
    st.chat_message("user" if r=="user" else "assistant").markdown(m)

# ================== INPUT ==================
user_input = st.chat_input("Ketik pesan atau ucapkan: Halo RANI")

if user_input and not st.session_state.processing:
    now = time.time()
    if now - st.session_state.last_message_time < 5:
        st.warning("⏳ Mohon tunggu beberapa detik.")
        st.stop()

    if not filter_spam(user_input):
        st.warning("⚠️ Pesan tidak valid.")
        st.stop()

    st.session_state.last_message_time = now
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.gesture = "think"
    st.session_state.processing = True
    st.rerun()

# ================== PROSES ==================
if st.session_state.processing and st.session_state.gesture == "think":
    q = st.session_state.chat_history[-1][1]
    ctx = cari_konteks(q)
    jawaban = jawab_gemini(q, ctx, st.session_state.chat_history)
    st.session_state.chat_history.append(("bot", jawaban))
    st.session_state.gesture = "speak"
    rani_bicara(jawaban)
    time.sleep(1.5)  # Beri waktu TTS mulai
    st.session_state.gesture = "idle"
    st.session_state.processing = False
    st.rerun()