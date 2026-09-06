# -*- coding: utf-8 -*-
import os
import re
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

st.title("🎬 Video Kể Chuyện Điện Ảnh 16:9")
st.caption("Khử rung Ken Burns, chống trùng ảnh và tối ưu file âm thanh dài")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn)", type="password", placeholder="Nhập key Pexels để tải ảnh thật siêu tốc")
audio_file = st.file_uploader("Tải lên file Voice / Âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def fetch_unique_media(query: str, idx: int, p_key: str, workdir: str, used_urls: set) -> str:
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False

    if p_key and p_key.strip():
        search_candidates = [
            query,
            query.split(",")[0],
            "student studying focus",
            "psychology thinking desk",
            "reading book library"
        ]
        for term in search_candidates:
            if downloaded:
                break
            try:
                page = (idx % 4) + 1
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(term)}&per_page=15&page={page}&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
                if r.ok and r.json().get("photos"):
                    photos = r.json()["photos"]
                    new_photos = [p for p in photos if p["src"]["large2x"] not in used_urls]
                    selected = new_photos[0] if new_photos else photos[0]

                    img_url = selected["src"]["large2x"]
                    used_urls.add(img_url)

                    img_data = requests.get(img_url, timeout=10).content
                    with open(dest, "wb") as f:
                        f.write(img_data)
                    downloaded = True
            except Exception:
                continue

    if not downloaded:
        try:
            lexica_url = f"https://lexica.art/api/v1/search?q={urllib.parse.quote(query + ' cinematic lighting realistic')}"
            r = requests.get(lexica_url, timeout=7)
            if r.ok and r.json().get("images"):
                images = r.json()["images"]
                new_images = [img for img in images if img["src"] not in used_urls]
                selected_img = new_images[0] if new_images else images[0]

                img_url = selected_img["src"]
                used_urls.add(img_url)

                img_data = requests.get(img_url, timeout=10).content
                with open(dest, "wb") as f:
                    f.write(img_data)
                downloaded = True
        except Exception:
            pass

    if not downloaded:
        img = Image.new('RGB', (W, H), color=(20, 24, 32))
        img.save(dest, "JPEG")
        return dest

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        img = Image.new('RGB', (W, H), color=(20, 24, 32))
        img.save(dest, "JPEG")

    return dest

def create_kenburns_clip(img_path: str, duration: float, out_clip: str, mode: int = 0):
    frames = max(25, int(duration * FPS))
    step = 0.18 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.18)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"if(eq(on,1),1.18,max(1.0,zoom-{step:.6f}))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_complex = (
        f"scale=2560:1440,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=2560x1440:fps={FPS},"
        f"scale={W}:{H}:flags=lanczos,"
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
        workdir = tempfile.mkdtemp(prefix="kb_fix_")
        used_urls = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh
            status.update(label="🎙️ 1/4: Đang phân tích lời thoại...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []

            # Gộp các câu quá ngắn để cảnh giữ tối thiểu 4-6s
            segments = []
            cur_text = ""
            cur_start = 0.0

            for seg in raw_segs:
                t = (seg.get("text") or "").strip()
                if not t:
                    continue
                if not cur_text:
                    cur_start = float(seg["start"])
                    cur_text = t
                else:
                    cur_text += " " + t

                # Mỗi cảnh tối thiểu 4.5s hoặc câu kết
                if float(seg["end"]) - cur_start >= 4.5:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "learning and education"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.0, total_audio_dur - segments[i]["start"])

            # 2. Phân tích kịch bản (Xử lý an toàn tránh vỡ JSON)
            status.update(label="🧠 2/4: AI lên danh sách hình ảnh phong phú...")
            by_idx = {}
            # Chia thành từng batch 15 cảnh nếu kịch bản quá dài
            batch_size = 15
            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:80]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Extract 1 specific English visual keyword (3-5 words) for each index.
Diverse subjects: student studying, classroom, library books, exam tension, writing on desk, clock ticking.
Lines:
{transcript_text}

Return strictly a JSON object:
{{"scenes": [{{"index": {b_start}, "search_query": "student writing exam paper"}}]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    content = llm_resp.choices[0].message.content
                    # Tìm khối JSON trong câu trả lời
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item.get("search_query", "")
                except Exception:
                    pass

            # 3. Gom ảnh và tạo clip Ken Burns
            status.update(label="🎥 3/4: Tải ảnh độc bản & tạo chuyển động Ken Burns...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    query = by_idx.get(idx) or "college student studying desk"
                    img_path = fetch_unique_media(query, idx, pexels_key, workdir, used_urls)
                    clip_out = os.path.join(workdir, f"clip_{idx:03d}.mp4")

                    create_kenburns_clip(img_path, sc["duration"], clip_out, mode=idx)
                    f_clips.write(f"file '{os.path.abspath(clip_out)}'\n")

            # 4. Xuất video hoàn thiện
            status.update(label="⚡ 4/4: Ghép video và đồng bộ âm thanh...")
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

            status.update(label="✅ Hoàn thành dựng video!", state="complete")

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
