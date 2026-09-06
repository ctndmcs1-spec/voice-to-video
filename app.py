# -*- coding: utf-8 -*-
import os
import json
import random
import shutil
import subprocess
import tempfile
import time
import urllib.parse

import streamlit as st
import requests
from PIL import Image, ImageOps
from groq import Groq

st.set_page_config(page_title="Voice to Video 16:9", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"

st.title("🎬 Tạo Video Biểu Cảm / Phân Cảnh (16:9)")
st.caption("Tối ưu tốc độ: Tải ảnh có sẵn siêu nhanh, không chờ vẽ AI")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn)", type="password", placeholder="Để trống để dùng ảnh minh họa có sẵn")

audio_file = st.file_uploader("Tải file âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def fetch_fast_media(query: str, idx: int, p_key: str, workdir: str) -> str:
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    # Ưu tiên 1: Pexels (nếu có key)
    if p_key and p_key.strip():
        for search_term in [query, query.split()[0] if query.split() else "abstract", "reaction"]:
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(search_term)}&per_page=5&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=5)
                if r.ok and r.json().get("photos"):
                    # Lấy ngẫu nhiên 1 trong các ảnh gần đúng
                    photos = r.json()["photos"]
                    img_url = random.choice(photos)["src"]["large2x"]
                    img_data = requests.get(img_url, timeout=10).content
                    with open(dest, "wb") as f:
                        f.write(img_data)
                    downloaded = True
                    break
            except Exception:
                continue

    # Ưu tiên 2: Tải ảnh minh họa / stickman / cartoon từ CDN có sẵn (mất 1-2s)
    if not downloaded:
        try:
            # Tìm ảnh qua Lexica API (Kho ảnh minh họa AI đã được render sẵn hàng triệu tấm, tải về ngay lập tức)
            lexica_url = f"https://lexica.art/api/v1/search?q={urllib.parse.quote(query + ' stickman cartoon illustration')}"
            r = requests.get(lexica_url, timeout=6)
            if r.ok and r.json().get("images"):
                img_url = r.json()["images"][0]["src"]
                img_data = requests.get(img_url, timeout=10).content
                with open(dest, "wb") as f:
                    f.write(img_data)
                downloaded = True
        except Exception:
            pass

    # Nếu tất cả đều rớt mạng: Dùng ảnh màu nền tối giản
    if not downloaded:
        img = Image.new('RGB', (W, H), color=(20, 25, 35))
        img.save(dest, "JPEG")
        return dest

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        img = Image.new('RGB', (W, H), color=(20, 25, 35))
        img.save(dest, "JPEG")

    return dest

if st.button("⚡ Bắt Đầu Tạo Video", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên!")
    else:
        status = st.status("Đang xử lý...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="fast_v2v_")
        
        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            client = Groq(api_key=groq_key.strip())

            # 1. Whisper
            status.update(label="🎙️ 1/4: Bóc tách giọng nói...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            segments = []
            for seg in (data.get("segments") or []):
                t = (seg.get("text") or "").strip()
                if t:
                    segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": t})

            if not segments:
                cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 5.0)
                segments.append({"start": 0.0, "end": dur, "text": "animation"})

            # 2. Tạo từ khóa biểu cảm đơn giản
            status.update(label="🧠 2/4: Lọc biểu cảm nhân vật...")
            transcript_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(segments)])
            prompt = f"""Phân tích các câu thoại sau và trích xuất cho MỖI đoạn 1 từ khóa biểu cảm cảm xúc (tiếng Anh ngắn gọn 1-2 từ, ví dụ: angry, funny stickman, shocked, running, happy):
{transcript_text}
Chỉ trả lời đúng định dạng JSON: {{"scenes": [{{"index": 0, "search_query": "shocked face"}}]}}"""

            llm_resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(llm_resp.choices[0].message.content).get("scenes", [])
            by_idx = {int(item.get("index", -1)): item.get("search_query", "") for item in parsed}

            # 3. Lấy ảnh tức thì (Chỉ mất 1-2s / ảnh)
            status.update(label="⚡ 3/4: Tải nhanh hình ảnh phù hợp...")
            concat_file = os.path.join(workdir, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as f_concat:
                for idx, sc in enumerate(segments):
                    query = by_idx.get(idx) or "stickman reaction"
                    img_path = fetch_fast_media(query, idx, pexels_key, workdir)
                    dur_scene = max(1.0, sc["end"] - sc["start"])
                    f_concat.write(f"file '{os.path.abspath(img_path)}'\n")
                    f_concat.write(f"duration {dur_scene:.2f}\n")

                last_img = os.path.join(workdir, f"media_{len(segments)-1:03d}.jpg")
                f_concat.write(f"file '{os.path.abspath(last_img)}'\n")

            # 4. Render video
            status.update(label="🎬 4/4: Ghép video MP4 16:9...")
            out_path = os.path.join(workdir, "output.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-i", audio_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                out_path
            ]
            subprocess.run(cmd, check=True)

            status.update(label="✅ Hoàn thành!", state="complete")
            
            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"video_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Lỗi: {str(e)}", state="error")
            st.error(f"Chi tiết: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
