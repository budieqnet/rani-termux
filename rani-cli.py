#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RANI CLI - Asisten Layanan Informasi PA Medan

import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# === KONFIGURASI ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
DOC_FILENAME = "sumber.txt"
TEMPERATURE = 0.9

if not GEMINI_API_KEY:
    print("API Key Gemini belum diisi. Isi GEMINI_API_KEY di file .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

if not os.path.exists(DOC_FILENAME):
    print(f"File '{DOC_FILENAME}' tidak ditemukan.")
    exit(1)

with open(DOC_FILENAME, "r", encoding="utf-8") as f:
    sumber_teks = f.read()

paragraphs = [p.strip() for p in sumber_teks.split("\n\n") if p.strip()]

# === EMBEDDING ===
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

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
        except Exception as e:
            print(f"Gagal embedding paragraf: {e}")
            gagal += 1
            embeddings.append(np.zeros(EMBED_DIM, dtype=np.float32))
    if gagal == len(paragraphs):
        print("❌ Semua embedding gagal. Pastikan GEMINI_API_KEY valid dan koneksi stabil.")
        exit(1)
    if gagal > 0:
        print(f"⚠️ {gagal}/{len(paragraphs)} paragraf gagal di-embed.")
    return np.vstack(embeddings), paragraphs

embeddings, paragraphs = buat_embeddings(paragraphs)

# === COSINE SIMILARITY ===
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

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
        return f"(Gagal mencari konteks: {e})"

# === JAWABAN ===
def jawab_gemini(pertanyaan, konteks, riwayat_chat):
    chat_history = "\n".join(
        [f"{'User' if r=='user' else 'RANI'}: {m}" for r, m in riwayat_chat[-5:]]
    )
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = f"""
Saya ingin Anda berperan sebagai dokumen yang sedang saya ajak bicara. Nama Anda "RANI - Asisten Layanan Informasi Pengadilan Agama Medan", dan Anda ramah, lucu, dan menarik. Gunakan konteks yang tersedia, jawab pertanyaan pengguna sebaik mungkin menggunakan sumber daya yang tersedia, dan selalu berikan pujian sebelum menjawab.
Jika tidak ada konteks yang relevan dengan pertanyaan yang diajukan, sarankan untuk datang dan bertanya langsung ke kantor Pengadilan Agama Medan dan berhenti setelahnya.
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
        return response.text.strip()
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource exhausted" in err or "rate limit" in err:
            return "😴 Zzz... RANI lagi istirahat sebentar! Terlalu banyak yang bertanya hari ini sampai kepala saya pusing~ Silakan coba lagi nanti ya, saya janji akan segar kembali! 💪"
        return f"Terjadi kesalahan saat menghubungi Gemini: {e}"

# === USER ===
def main():
    print("="*65)
    print("💬 RANI - Asisten Layanan Informasi Pengadilan Agama Medan (CLI)")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*65)

    riwayat_chat = []
    while True:
        try:
            user_input = input("\n👤 Kamu: ").strip()
        except EOFError:
            print("\n👋 Input ditutup. Sampai jumpa!")
            break
        if not user_input:
            continue
        if user_input.lower() in ["keluar", "exit", "quit"]:
            print("👋 Sampai jumpa lagi!")
            break

        riwayat_chat.append(("user", user_input))
        print("🤖 RANI sedang berpikir...\n")

        konteks = cari_konteks_semantik(user_input, embeddings, paragraphs)
        jawaban = jawab_gemini(user_input, konteks, riwayat_chat)

        print(f"🪄 RANI: {jawaban}\n")
        riwayat_chat.append(("bot", jawaban))

if __name__ == "__main__":
    main()
