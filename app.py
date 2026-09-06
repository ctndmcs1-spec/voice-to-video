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

st.title("🎬 Video Kể Chuyện Điện Ảnh (Ken Burns Effect)")
st.caption("Khắc phục lệch tiếng, tự động Zoom/Pan điện ảnh cho từng khung hình")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn)", type="password", placeholder="Nhập key Pexels để tải ảnh thật siêu tốc")
audio_file = st.file_uploader("Tải lên file Voice / Âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def fetch_story_media(query: str, idx: int, p_key: str, workdir: str) -> str:
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    # 1. Thử lấy Pexels nếu có key
    if p_key and p_key.strip():
        for term in [query, query.split(",")[0], "mystery dramatic"]:
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

    # 2. Lấy kho Lexica ảnh vẽ cinematic chuẩn kịch bản
    if not downloaded:
        try:
            lexica_url = f"https://lexica.art/api/v1/search?q={urllib.parse.quote(query + ' cinematic lighting, 8k, photorealistic')}"
            r = requests.get(lexica_url, timeout=7)
            if r.ok and r.json().get("images"):
                img_url = r.json()["images"][0]["src"]
                img_data = requests.get(img_url, timeout=10).content
                with open(dest, "wb") as f:
                    f.write(img_data)
                downloaded = True
        except Exception:
            pass

    # 3. Fallback màu nền nếu rớt mạng
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

def create_kenburns_clip(img_path: str, duration: float, out_clip: str, mode: int = 0):
    """Tạo clip ngắn có hiệu ứng Zoom In / Zoom Out từ ảnh tĩnh"""
    frames = max(1, int(duration * FPS))
    # Chuyển đổi qua lại giữa Zoom In và Zoom Out
    if mode % 2 == 0:
        # Zoom In từ từ vào tâm
        z_expr = f"min(zoom+0.0012,1.25)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        # Zoom Out từ 1.25 về 1.0
        z_expr = f"if(eq(on,1),1.25,max(1.0,zoom-0.0012))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_complex = (
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={W}x{H}:fps={FPS},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", filter_complex,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

if st.button("⚡ Bắt Đầu Dựng Video", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên trước!")
    else:
        status = st.status("Đang khởi động tiến trình dựng phim...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="kb_v2v_")
        
        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Đo độ dài chính xác của toàn bộ file audio
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Whisper bóc tách mốc thời gian
            status.update(label="🎙️ 1/4: Đo đạc mốc thời gian chính xác từng câu thoại...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []

            segments = []
            for seg in raw_segs:
                t = (seg.get("text") or "").strip()
                if t:
                    segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": t})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "mysterious dramatic story"})

            # ĐỒNG BỘ THỜI LƯỢNG (Fix lỗi tiếng chạy trước hình):
            # Thời lượng cảnh i kéo dài cho tới khi cảnh i+1 bắt đầu (bao trùm cả khoảng lặng)
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(1.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(1.0, total_audio_dur - segments[i]["start"])

            # 2. LLM chọn bối cảnh kịch tính
            status.update(label="🧠 2/4: AI đạo diễn phân tích biểu cảm & bối cảnh...")
            transcript_text = "\n".join([f"[{i}] {s['text']}" for i, s in enumerate(segments)])
            prompt = f"""
            Bạn là đạo diễn hình ảnh cho truyện trinh thám/tâm lý kịch tính.
            Dựa trên kịch bản sau, trích xuất cho MỖI đoạn 1 mô tả hình ảnh tiếng Anh thật cụ thể (3-6 từ), tập trung vào NHÂN VẬT, BÁC SĨ, VĂN PHÒNG, BIỂU CẢM:
            {transcript_text}

            Chỉ trả về JSON thuần túy theo định dạng:
            {{"scenes": [{{"index": 0, "search_query": "psychiatrist listening intently in dimly lit clinic"}}]}}
            """

            llm_resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(llm_resp.choices[0].message.content).get("scenes", [])
            by_idx = {int(item.get("index", -1)): item.get("search_query", "") for item in parsed}

            # 3. Tải ảnh và tạo các clip Ken Burns có chuyển động
            status.update(label="🎥 3/4: Đang tạo hiệu ứng Zoom In / Zoom Out cho từng cảnh...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    query = by_idx.get(idx) or "mysterious dramatic psychological scene"
                    img_path = fetch_story_media(query, idx, pexels_key, workdir)
                    clip_out = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                    
                    # Tạo hiệu ứng chuyển động Ken Burns
                    create_kenburns_clip(img_path, sc["duration"], clip_out, mode=idx)
                    f_clips.write(f"file '{os.path.abspath(clip_out)}'\n")

            # 4. Ghép toàn bộ clip lại cùng audio gốc
            status.update(label="⚡ 4/4: Ghép video và đồng bộ âm thanh hoàn chỉnh...")
            out_path = os.path.join(workdir, "output.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
            subprocess.run(cmd, check=True)

            status.update(label="✅ Đã dựng xong video hoàn chỉnh!", state="complete")
            
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
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
