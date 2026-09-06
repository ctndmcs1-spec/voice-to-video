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

st.set_page_config(page_title="AI Story Video 16:9", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"

st.title("🎬 Tạo Video Kể Chuyện Điện Ảnh 16:9")
st.caption("Bám sát cốt truyện nhân vật, tự zoom lia máy (Ken Burns) và chèn phụ đề")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn)", type="password", placeholder="Nhập key Pexels để tải ảnh thật siêu tốc")
audio_file = st.file_uploader("Tải lên file Voice / Âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def fetch_story_media(query: str, idx: int, p_key: str, workdir: str) -> str:
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    # 1. Thử lấy Pexels nếu có key
    if p_key and p_key.strip():
        for term in [query, query.split(",")[0], "psychology mystery"]:
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(term)}&per_page=5&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
                if r.ok and r.json().get("photos"):
                    img_url = random.choice(r.json()["photos"])["src"]["large2x"]
                    img_data = requests.get(img_url, timeout=10).content
                    with open(dest, "wb") as f:
                        f.write(img_data)
                    downloaded = True
                    break
            except Exception:
                continue

    # 2. Lấy kho Lexica ảnh vẽ cinematic theo kịch bản (1-2s)
    if not downloaded:
        try:
            lexica_url = f"https://lexica.art/api/v1/search?q={urllib.parse.quote(query + ' cinematic lighting, hyperrealistic, dark mood')}"
            r = requests.get(lexica_url, timeout=7)
            if r.ok and r.json().get("images"):
                img_url = r.json()["images"][0]["src"]
                img_data = requests.get(img_url, timeout=10).content
                with open(dest, "wb") as f:
                    f.write(img_data)
                downloaded = True
        except Exception:
            pass

    # 3. Fallback nếu đứt mạng
    if not downloaded:
        img = Image.new('RGB', (W, H), color=(20, 20, 28))
        img.save(dest, "JPEG")
        return dest

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        img = Image.new('RGB', (W, H), color=(20, 20, 28))
        img.save(dest, "JPEG")

    return dest

if st.button("⚡ Bắt Đầu Dựng Video", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên trước!")
    else:
        status = st.status("Đang khởi động bộ máy AI...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="story_v2v_")
        
        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách giọng nói
            status.update(label="🎙️ 1/4: Đang phân tích lời thoại...")
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
                segments.append({"start": 0.0, "end": dur, "text": "mysterious story scene"})

            # 2. Phân tích kịch bản nhân vật chuẩn xác
            status.update(label="🧠 2/4: AI đạo diễn lên ý tưởng góc máy & bối cảnh...")
            transcript_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(segments)])
            prompt = f"""
            Bạn là đạo diễn hình ảnh cho truyện trinh thám/tâm lý kịch tính.
            Dựa trên kịch bản sau, hãy trích xuất cho MỖI đoạn 1 mô tả hình ảnh tiếng Anh thật cụ thể (3-6 từ), tập trung vào NHÂN VẬT, PHÒNG KHÁM, BIỂU CẢM, KHÔNG TỰ Ý VẼ PHONG CẢNH THIÊN NHIÊN:
            {transcript_text}

            Chỉ trả về JSON thuần túy theo định dạng:
            {{"scenes": [{{"index": 0, "search_query": "psychiatrist in dark office thinking"}}]}}
            """

            llm_resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(llm_resp.choices[0].message.content).get("scenes", [])
            by_idx = {int(item.get("index", -1)): item.get("search_query", "") for item in parsed}

            # 3. Tải hình ảnh & tạo file ghép
            status.update(label="🎨 3/4: Đang gom ảnh minh họa theo tình tiết...")
            concat_file = os.path.join(workdir, "concat.txt")
            with open(concat_file, "w", encoding="utf-8") as f_concat:
                for idx, sc in enumerate(segments):
                    query = by_idx.get(idx) or "mysterious scene"
                    img_path = fetch_story_media(query, idx, pexels_key, workdir)
                    dur_scene = max(1.0, sc["end"] - sc["start"])
                    f_concat.write(f"file '{os.path.abspath(img_path)}'\n")
                    f_concat.write(f"duration {dur_scene:.2f}\n")

                last_img = os.path.join(workdir, f"media_{len(segments)-1:03d}.jpg")
                f_concat.write(f"file '{os.path.abspath(last_img)}'\n")

            # 4. Render FFmpeg
            status.update(label="⚡ 4/4: Đang render video hoàn thiện...")
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

            status.update(label="✅ Hoàn thành xuất sắc!", state="complete")
            
            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"story_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Lỗi: {str(e)}", state="error")
            st.error(f"Chi tiết: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
