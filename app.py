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

st.title("🎬 Multi-Genre POV Studio Engine")
st.caption("Tỷ lệ 40% Video B-roll + 60% Ảnh tĩnh, đa dạng Tone cảm xúc, hình ảnh sát voice 100%")

# GIAO DIỆN CHỌN TONE PHONG CÁCH
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
pexels_key = st.text_input("Pexels API Key (Để lấy video B-roll 40% HD)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh", type=["mp3", "wav", "m4a", "ogg"])

# ==============================================================================
# HỆ THỐNG PHỦ MÀU COLOR GRADING THEO TONE PHONG CÁCH
# ==============================================================================
def apply_genre_color_grading(img: Image.Image, genre: str) -> Image.Image:
    """Tự động đổi tông màu sắc tương ứng theo phong cách được chọn"""
    if "Bright Career" in genre:
        # Tươi sáng, ấm áp, rực rỡ
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.06)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.20)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.10)
        return img

    elif "Dark Moody" in genre:
        # U tối, tương phản gắt, phủ vignette đen 4 góc
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
        # Ánh xanh công nghệ, sắc sảo, tương phản cao
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.20)
        r, g, b = graded.split()
        b = b.point(lambda i: min(255, int(i * 1.08)))
        return Image.merge("RGB", (r, g, b))

    else: # Vintage Lofi
        # Ấm vàng hoài cổ, giảm tương phản nhẹ
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(0.95)
        r, g, b = graded.split()
        r = r.point(lambda i: min(255, int(i * 1.06)))
        return Image.merge("RGB", (r, g, b))

# ==============================================================================
# BỘ TÌM KIẾM ẢNH SÁT NGHĨA SONG NGỮ (VIỆT + NGOẠI)
# ==============================================================================
def fetch_matching_image(query_vn: str, query_en: str, idx: int, workdir: str, used_urls: set, p_key: str, genre: str) -> str:
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. Tìm ảnh Việt Nam thực tế
    vn_terms = [f"{query_vn} chụp thực tế", query_vn]
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
                                    if t_img.size[0] >= 600:
                                        used_urls.add(u)
                                        downloaded = True
                                        break
                        except Exception:
                            continue
        except Exception:
            continue

    # 2. Tìm ảnh kho ngoại Pexels chất lượng cao
    if not downloaded and p_key and p_key.strip():
        en_terms = [query_en, f"{query_en} authentic"]
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

    # 3. Fallback an toàn
    if not downloaded:
        fb_term = query_en if query_en else "daily work activity"
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
        img = Image.new('RGB', (W, H), color=(40, 45, 55))
        img.save(dest, "JPEG")

    try:
        with Image.open(dest) as raw_img:
            fitted = ImageOps.fit(raw_img.convert("RGB"), (W, H), Image.LANCZOS)
            graded = apply_genre_color_grading(fitted, genre)
            graded.save(dest, "JPEG", quality=92)
    except Exception:
        pass

    return dest

# ==============================================================================
# BỘ LẤY VIDEO B-ROLL (ĐẠT TỶ LỆ 40%, KHÔNG TRÙNG LẶP)
# ==============================================================================
def fetch_broll_clip(query_en: str, idx: int, duration: float, p_key: str, workdir: str, used_vid_ids: set) -> str:
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False

    search_terms = [query_en, f"{query_en} action", "person working close up", "city street traffic"]

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
    """Zoom/Lia máy 2K mượt mà cho 60% cảnh ảnh tĩnh"""
    frames = max(25, int(duration * FPS))
    step = 0.16 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.16)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"if(eq(on,1),1.16,max(1.0,zoom-{step:.6f}))"
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

# ==============================================================================
# PIPELINE ĐIỀU PHỐI CHÍNH
# ==============================================================================
if st.button("⚡ Bắt Đầu Dựng Video Đa Phong Cách (40% Video + 60% Ảnh)", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        status = st.status("Đang kích hoạt cỗ máy dựng video đa phong cách...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="multi_pov_")
        used_urls = set()
        used_vid_ids = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian câu thoại...")
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

                if float(seg["end"]) - cur_start >= 4.2:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "câu chuyện đời sống"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.0, total_audio_dur - segments[i]["start"])

            # 2. AI Đạo diễn bóc tách từ khóa hành động sát voice theo Tone
            status.update(label=f"🧠 2/4: AI đạo diễn trích xuất hình ảnh sát nghĩa theo phong cách {genre_mode}...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn hình ảnh cho video phong cách: "{genre_mode}".
Nhiệm vụ: Trích xuất cho MỖI câu thoại một hành động/vật thể cụ thể SÁT VỚI NỘI DUNG NÓI (3-5 từ), gồm cả tiếng Việt và tiếng Anh.
QUY TẮC:
- Nếu phong cách là Bright Career (Nghề nghiệp tươi sáng): chọn hành động làm việc tích cực, tay nghề, dụng cụ, quán xá, nụ cười (ví dụ: 'pha cà phê espresso', 'thợ làm bánh nhào bột', 'nhân viên thu ngân tính tiền', 'bàn làm việc sáng sủa').
- Nếu phong cách là Dark Moody (Tâm lý u tối): chọn bối cảnh trầm ngâm, góc tối, ánh sáng hắt, phòng vắng.
- CẤM ruộng lúa, làng quê trừ khi lời thoại nhắc trực tiếp đến đồng quê.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "pha cà phê latte art", "query_en": "barista making coffee"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.25
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Gom tài nguyên theo tỷ lệ 40% Video B-roll + 60% Ảnh
            status.update(label="🎬 3/4: Đang ghép 40% Video B-roll + 60% Ảnh tĩnh Color Graded...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "hoạt động đời sống"
                    query_en = sc_data.get("query_en") or "daily work life"
                    dur = sc["duration"]

                    clip_path = None
                    # TỶ LỆ 40% VIDEO: Cảnh thứ 2 và thứ 4 trong mỗi chu kỳ 5 cảnh (idx % 5 in [1, 3]) sẽ lấy video clip
                    is_video_slot = (idx % 5 in [1, 3])
                    if is_video_slot and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, dur, pexels_key, workdir, used_vid_ids)

                    # 60% ẢNH TĨNH: Chạy Ken Burns 2K với Color Grading đúng Tone
                    if not clip_path:
                        img_path = fetch_matching_image(query_vn, query_en, idx, workdir, used_urls, pexels_key, genre_mode)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, dur, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video hoàn thiện
            status.update(label="⚡ 4/4: Nối các cảnh và đồng bộ âm thanh hoàn chỉnh...")
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

            status.update(label="✅ Video đa phong cách hoàn thiện xuất sắc!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Hoàn Chỉnh Về Máy",
                data=video_bytes,
                file_name=f"multi_pov_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
