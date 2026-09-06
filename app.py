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

st.set_page_config(page_title="AI Voice to Video 16:9", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width=1280&height=720&nologo=true&seed={seed}"

st.title("🎬 Tạo Video 16:9 Tự Động")
st.caption("Bóc tách giọng nói, phân tích cảnh và ghép video MP4 chuẩn YouTube")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn, để trống dùng ảnh AI)", type="password", placeholder="Để trống nếu muốn AI tự tạo ảnh")

audio_file = st.file_uploader("Tải lên file Voice / Âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def fetch_media(query: str, idx: int, p_key: str, workdir: str) -> str:
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    # 1. Thử lấy ảnh thật từ Pexels (nếu có key)
    if p_key and p_key.strip():
        try:
            url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(query)}&per_page=1&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=15)
            if r.ok and r.json().get("photos"):
                img_url = r.json()["photos"][0]["src"]["large2x"]
                img_data = requests.get(img_url, timeout=20).content
                with open(dest, "wb") as f:
                    f.write(img_data)
                downloaded = True
        except Exception:
            downloaded = False

    # 2. Nếu không có Pexels, dùng AI vẽ ảnh
    if not downloaded:
        try:
            safe_q = urllib.parse.quote(f"cinematic 16:9 landscape photography, {query}"[:160])
            ai_url = POLLINATIONS_URL.format(prompt=safe_q, seed=random.randint(1, 99999))
            # Tăng timeout lên 90s để chống sập khi server load chậm
            img_data = requests.get(ai_url, timeout=90).content
            with open(dest, "wb") as f:
                f.write(img_data)
        except Exception as e:
            # LỚP BẢO VỆ CHỐNG SẬP: Nếu vẽ ảnh vẫn lỗi, tạo ảnh đen có chữ để video đi tiếp
            img = Image.new('RGB', (W, H), color=(30, 30, 30))
            img.save(dest, "JPEG")
            return dest

    # Resize ảnh cho đúng chuẩn 16:9
    with Image.open(dest) as img:
        fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
        fitted.save(dest, "JPEG", quality=90)
    return dest

if st.button("⚡ Bắt Đầu Tạo Video", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên trước!")
    else:
        status = st.status("Đang khởi động tiến trình...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="s2v_")
        
        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            client = Groq(api_key=groq_key.strip())

            status.update(label="🎙️ Bước 1/4: Đang bóc tách lời thoại qua Whisper...")
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
                segments.append({"start": 0.0, "end": dur, "text": data.get("text", "scenic landscape")})

            status.update(label="🧠 Bước 2/4: AI đang phân tích nội dung và chọn bối cảnh...")
            transcript_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(segments)])
            prompt = f"""Phân tích các đoạn kịch bản sau và trích xuất cho MỖI đoạn 1 từ khóa tiếng Anh (2-4 từ, phong cảnh 16:9 cinematic):
{transcript_text}
Chỉ trả về JSON thuần túy theo định dạng: {{"scenes": [{{"index": 0, "search_query": "concrete visual keyword"}}]}}"""

            llm_resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(llm_resp.choices[0].message.content).get("scenes", [])
            by_idx = {int(item.get("index", -1)): item.get("search_query", "") for item in parsed}

            status.update(label="🖼️ Bước 3/4: Đang tải và xử lý khung hình 16:9 (Có thể mất 1-2 phút)...")
            concat_file = os.path.join(workdir, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as f_concat:
                for idx, sc in enumerate(segments):
                    query = by_idx.get(idx) or "scenic nature landscape"
                    img_path = fetch_media(query, idx, pexels_key, workdir)
                    dur_scene = max(1.0, sc["end"] - sc["start"])
                    f_concat.write(f"file '{os.path.abspath(img_path)}'\n")
                    f_concat.write(f"duration {dur_scene:.2f}\n")

                last_img = os.path.join(workdir, f"media_{len(segments)-1:03d}.jpg")
                f_concat.write(f"file '{os.path.abspath(last_img)}'\n")

            status.update(label="⚡ Bước 4/4: Đang xuất video hoàn chỉnh...")
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

            status.update(label="✅ Đã tạo video thành công!", state="complete")
            
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
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
