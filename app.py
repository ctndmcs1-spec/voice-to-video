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
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Multi-Genre POV Video Engine", page_icon="🎬", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🎬 Multi-Genre POV Studio Engine (Sync Master)")
st.caption("Khớp âm thanh chuẩn xác 100%, chống màn đen cuối video, đan xen 40% Video B-roll + 60% Ảnh")

genre_mode = st.selectbox(
    "Chọn phong cách & Tone màu chủ đạo của Video:",
    [
        "Nghề nghiệp / Tươi sáng & Động lực (Bright Career)",
        "Tâm lý / Góc khuất & U tối (Dark Moody POV)",
        "Tài chính / Khởi nghiệp & Kịch tính (Corporate / Hustle)",
        "Đời sống thường nhật & Hoài niệm (Vintage Lofi Life)"
    ]
)

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Để lấy video B-roll HD)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh", type=["mp3", "wav", "m4a", "ogg"])

def apply_genre_color_grading(img: Image.Image, genre: str) -> Image.Image:
    if "Bright Career" in genre:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.06)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.20)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.10)
        return img
    elif "Dark Moody" in genre:
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.25)
        enhancer = ImageEnhance.Brightness(graded)
        graded = enhancer.enhance(0.85)
        vignette = Image.new("L", (W, H), 255)
        d_v = ImageDraw.Draw(vignette)
        d_v.ellipse([-W * 0.15, -H * 0.15, W * 1.15, H * 1.15], fill=0)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=110))
        black_layer = Image.new("RGB", (W, H), (10, 12, 16))
        return Image.composite(black_layer, graded, vignette)
    elif "Corporate" in genre:
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.20)
        r, g, b = graded.split()
        b = b.point(lambda i: min(255, int(i * 1.08)))
        return Image.merge("RGB", (r, g, b))
    else:
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(0.95)
        r, g, b = graded.split()
        r = r.point(lambda i: min(255, int(i * 1.06)))
        return Image.merge("RGB", (r, g, b))

def fetch_matching_image(query_vn: str, query_en: str, idx: int, workdir: str, used_urls: set, p_key: str, genre: str) -> str:
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    vn_terms = [f"{query_vn} thực tế", query_vn]
    for q in vn_terms:
        if downloaded:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, region="vn-vi", max_results=6))
                for r in results:
                    u = r.get("image")
                    if u and u.startswith("http") and u not in used_urls:
                        try:
                            resp = requests.get(u, headers=headers, timeout=5)
                            if resp.status_code == 200 and len(resp.content) > 30000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                with Image.open(dest) as t_img:
                                    if t_img.size[0] >= 500:
                                        used_urls.add(u)
                                        downloaded = True
                                        break
                        except Exception:
                            continue
        except Exception:
            continue

    if not downloaded and p_key and p_key.strip():
        en_terms = [query_en, f"{query_en} cinematic"]
        for q in en_terms:
            if downloaded:
                break
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(q)}&per_page=6&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
                if r.ok and r.json().get("photos"):
                    for p in r.json()["photos"]:
                        u = p["src"]["large2x"]
                        if u not in used_urls:
                            used_urls.add(u)
                            with open(dest, "wb") as f:
                                f.write(requests.get(u, timeout=8).content)
                            downloaded = True
                            break
            except Exception:
                continue

    if not downloaded:
        fb_term = query_en if query_en else "modern office work"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(fb_term, max_results=4))
                for r in results:
                    u = r.get("image")
                    if u and u.startswith("http") and u not in used_urls:
                        resp = requests.get(u, headers=headers, timeout=5)
                        if resp.status_code == 200 and len(resp.content) > 20000:
                            with open(dest, "wb") as f:
                                f.write(resp.content)
                            downloaded = True
                            break
        except Exception:
            pass

    if not downloaded:
        img = Image.new('RGB', (W, H), color=(30, 35, 45))
        img.save(dest, "JPEG")

    try:
        with Image.open(dest) as raw_img:
            fitted = ImageOps.fit(raw_img.convert("RGB"), (W, H), Image.LANCZOS)
            graded = apply_genre_color_grading(fitted, genre)
            graded.save(dest, "JPEG", quality=92)
    except Exception:
        pass

    return dest

def fetch_broll_clip(query_en: str, idx: int, duration: float, p_key: str, workdir: str, used_vid_ids: set) -> str:
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False

    search_terms = [query_en, f"{query_en} action", "office professional working", "city night building"]

    for term in search_terms:
        if downloaded:
            break
        try:
            url = f"{PEXELS_VIDEO_URL}?query={urllib.parse.quote(term)}&per_page=6&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
            if r.ok and r.json().get("videos"):
                for v in r.json()["videos"]:
                    v_id = v.get("id")
                    if v_id and v_id not in used_vid_ids:
                        used_vid_ids.add(v_id)
                        vid_files = v.get("video_files", [])
                        hd_files = [f for f in vid_files if f.get("height", 0) >= 720 and f.get("file_type") == "video/mp4"]
                        target_url = hd_files[0]["link"] if hd_files else vid_files[0]["link"]
                        with requests.get(target_url, stream=True, timeout=15) as stream:
                            with open(raw_vid, "wb") as f_out:
                                shutil.copyfileobj(stream.raw, f_out)
                        downloaded = True
                        break
        except Exception:
            continue

    if downloaded and os.path.exists(raw_vid):
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
    frames = max(25, int(duration * FPS))
    step = 0.15 / frames
    m = mode % 4

    if m == 0:
        z_expr = f"min(zoom+{step:.6f},1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif m == 1:
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*(on/{frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif m == 2:
        z_expr = f"if(eq(on,1),1.15,max(1.0,zoom-{step:.6f}))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*(1-on/{frames})"
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

if st.button("⚡ Bắt Đầu Dựng Video Chuẩn Xác Tuyệt Đối", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        status = st.status("Đang khởi động cỗ máy dựng video đồng bộ...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="sync_pov_")
        used_urls = set()
        used_vid_ids = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Đo độ dài file audio chính xác
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian chi tiết...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []

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
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "câu chuyện đời sống"})

            # Khóa thời lượng từng cảnh khớp 100% với tổng thời gian audio
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(1.5, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(1.5, total_audio_dur - segments[i]["start"])

            # 2. AI Đạo diễn bóc tách từ khóa
            status.update(label="🧠 2/4: AI đạo diễn trích xuất bối cảnh sát câu chữ...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn hình ảnh cho video phong cách: "{genre_mode}".
Nhiệm vụ: Trích xuất cho MỖI câu thoại một hành động/vật thể cụ thể SÁT TỪNG CHỮ VỚI NỘI DUNG NÓI (3-5 từ), gồm cả tiếng Việt và tiếng Anh.
QUY TẮC CỐT LÕI:
- CẢNH ĐẦU TIÊN [0]: Bắt buộc bám sát từng từ của câu mở đầu (nếu nhắc tòa nhà chọc trời, thành phố đêm -> 'tòa nhà chọc trời ban đêm', 'skyscraper night lights city'). Tuyệt đối không lấy ý các câu sau đè lên câu đầu.
- Các cảnh sau: Trích xuất sát hành động cụ thể (kiểm toán, máy tính, bảng số liệu, cà phê, ký tài liệu, họp hành).
- CẤM ruộng lúa, làng quê.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "tòa nhà chọc trời ban đêm", "query_en": "skyscraper night lights"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Dựng cảnh đan xen 40% Video + 60% Ảnh
            status.update(label="🎬 3/4: Đang ghép 40% Video B-roll + 60% Ảnh Ken Burns đa hướng...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "hoạt động làm việc"
                    query_en = sc_data.get("query_en") or "modern workplace"
                    dur = sc["duration"]

                    clip_path = None
                    # 40% Video: Phân bổ vào các cảnh có chỉ số chia dư cho 5 là 1 và 3
                    is_video_slot = (idx % 5 in [1, 3])
                    if is_video_slot and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, dur, pexels_key, workdir, used_vid_ids)

                    # 60% Ảnh tĩnh: Chạy Ken Burns đa trục
                    if not clip_path:
                        img_path = fetch_matching_image(query_vn, query_en, idx, workdir, used_urls, pexels_key, genre_mode)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, dur, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video hoàn thiện (Đồng bộ triệt để)
            status.update(label="⚡ 4/4: Ghép video và đồng bộ audio chuẩn xác...", state="running")
            out_path = os.path.join(workdir, "output.mp4")
            
            # Khử triệt để lỗi loop audio và màn đen cuối video bằng re-encode âm thanh chuẩn AAC
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
            subprocess.run(cmd, check=True)

            status.update(label="✅ Video hoàn thành chuẩn xác, đồng bộ hoàn hảo!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Hoàn Chỉnh Về Máy",
                data=video_bytes,
                file_name=f"synced_pov_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
