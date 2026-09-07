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
from duckduckgo_search import DDGS

st.set_page_config(page_title="Vietnam POV Video Engine", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🇻🇳 Dựng Video Kể Chuyện Thuần Việt (Ảnh + Video B-Roll)")
st.caption("Cào ảnh tư liệu đời sống thực tế Việt Nam, xen kẽ video chuyển động 5s")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn - Dùng để lấy video B-roll HD)", type="password", placeholder="Nhập key để lấy video stock chuyển động...")
audio_file = st.file_uploader("Tải lên file Voice / Âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def crawl_vietnam_image(query_vn: str, idx: int, workdir: str, used_urls: set) -> str:
    """Cào ảnh đời thực từ mạng bằng từ khóa tiếng Việt thuần túy"""
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    # Danh sách từ khóa tìm kiếm tiếng Việt ưu tiên thực tế
    search_queries = [
        f"{query_vn} đời sống thực tế việt nam",
        query_vn,
        "phòng trọ chung cư bình dân việt nam",
        "đường phố xe máy việt nam"
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for q in search_queries:
        if downloaded:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, region="vn-vi", max_results=8))
                for r in results:
                    img_url = r.get("image")
                    if img_url and img_url not in used_urls and img_url.startswith("http"):
                        try:
                            resp = requests.get(img_url, headers=headers, timeout=6)
                            if resp.status_code == 200 and len(resp.content) > 10000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                used_urls.add(img_url)
                                downloaded = True
                                break
                        except Exception:
                            continue
        except Exception:
            continue

    if not downloaded:
        # Fallback nền màu tối nếu rớt mạng
        img = Image.new('RGB', (W, H), color=(30, 32, 40))
        img.save(dest, "JPEG")
        return dest

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        img = Image.new('RGB', (W, H), color=(30, 32, 40))
        img.save(dest, "JPEG")

    return dest

def fetch_broll_clip(query_en: str, idx: int, duration: float, p_key: str, workdir: str) -> str:
    """Lấy video clip chuyển động 5s cắt chuẩn 1280x720 không tiếng"""
    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False

    if p_key and p_key.strip():
        try:
            url = f"{PEXELS_VIDEO_URL}?query={urllib.parse.quote(query_en)}&per_page=5&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
            if r.ok and r.json().get("videos"):
                videos = r.json()["videos"]
                vid_files = videos[0].get("video_files", [])
                hd_files = [f for f in vid_files if f.get("height", 0) >= 720]
                target_url = hd_files[0]["link"] if hd_files else vid_files[0]["link"]
                
                with requests.get(target_url, stream=True, timeout=15) as stream:
                    with open(raw_vid, "wb") as f_out:
                        shutil.copyfileobj(stream.raw, f_out)
                downloaded = True
        except Exception:
            pass

    if downloaded and os.path.exists(raw_vid):
        # Cắt đúng thời lượng duration (hoặc tối đa 5s), bỏ tiếng và ép về 1280x720
        filter_str = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p"
        cmd = [
            "ffmpeg", "-y", "-ss", "0", "-i", raw_vid,
            "-t", f"{duration:.3f}",
            "-vf", filter_str,
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast",
            clip_dest
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(raw_vid):
            os.remove(raw_vid)
        return clip_dest

    return None

def create_kenburns_clip(img_path: str, duration: float, out_clip: str, mode: int = 0):
    """Tạo chuyển động lia/zoom nhẹ mượt mà từ ảnh tĩnh cào được"""
    frames = max(25, int(duration * FPS))
    step = 0.15 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"if(eq(on,1),1.15,max(1.0,zoom-{step:.6f}))"
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

if st.button("⚡ Bắt Đầu Tạo Video Thuần Việt", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên trước!")
    else:
        status = st.status("Đang chạy tiến trình dựng phim...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="vn_story_")
        used_urls = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh
            status.update(label="🎙️ 1/4: Whisper bóc tách mốc thời gian từng câu thoại...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []

            # Gộp thành các cảnh khoảng 4-5s để đủ nhịp nhìn
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

                if float(seg["end"]) - cur_start >= 4.5:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "đời sống thường nhật"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.0, total_audio_dur - segments[i]["start"])

            # 2. AI bóc tách từ khóa thuần Việt
            status.update(label="🧠 2/4: AI trích xuất từ khóa bối cảnh đời sống Việt Nam...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:80]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Trích xuất cho MỖI đoạn 1 cụm từ tìm kiếm hình ảnh thuần Việt thực tế (tiếng Việt ngắn gọn 3-5 từ, ví dụ: phòng trọ bừa bộn, chung cư cũ hà nội, quầy tính tiền siêu thị, đường phố xe máy, bồn rửa bát ngập đĩa).
Đoạn thoại:
{transcript_text}

Trả về định dạng JSON:
{{"scenes": [{{"index": {b_start}, "query_vn": "phòng trọ bừa bộn đồ đạc", "query_en": "messy room apartment"}}]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Gom tài nguyên: Đan xen Ảnh thuần Việt & Video B-roll
            status.update(label="🎬 3/4: Đang cào ảnh mạng Việt Nam và ghép xen kẽ video B-roll...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "đời sống việt nam"
                    query_en = sc_data.get("query_en") or "city street walking"
                    dur = sc["duration"]

                    clip_path = None
                    # Cứ cảnh thứ 3 (idx % 3 == 2) thì thử lấy video B-roll chuyển động
                    if idx % 3 == 2 and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, dur, pexels_key, workdir)

                    # Nếu không phải cảnh video hoặc tải video thất bại: Cào ảnh Việt Nam và zoom Ken Burns
                    if not clip_path:
                        img_path = crawl_vietnam_image(query_vn, idx, workdir, used_urls)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, dur, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video hoàn thiện
            status.update(label="⚡ 4/4: Nối các cảnh và đồng bộ âm thanh...")
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

            status.update(label="✅ Đã dựng xong video thuần Việt!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"vietnam_story_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Lỗi: {str(e)}", state="error")
            st.error(f"Chi tiết: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
