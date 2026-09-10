# -*- coding: utf-8 -*-
import os
import re
import json
import math
import random
import shutil
import hashlib
import subprocess
import tempfile
import time
import urllib.parse

import streamlit as st
import requests
from PIL import Image, ImageOps
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Studio POV Master Engine - Voice Match", page_icon="🎬", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

# Chống trùng qua session
if "used_img_hashes" not in st.session_state:
    st.session_state.used_img_hashes = set()
if "used_vid_ids" not in st.session_state:
    st.session_state.used_vid_ids = set()

st.title("🎬 Studio POV Master Engine Pro")
st.caption("Bám sát 100% ngữ cảnh voice • Tự động nhận diện • Chống lệch đề & trùng lặp")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Khuyến nghị - để có video + ảnh chất lượng)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

# ==============================================================================
# HÀM PHỤ TRỢ
# ==============================================================================
def download_file(url: str, dest: str, headers=None, min_size: int = 30000, timeout: int = 8) -> bool:
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=timeout, stream=True)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)
        return os.path.getsize(dest) >= min_size
    except Exception:
        return False


def image_content_hash(path: str) -> str:
    """Hash nội dung ảnh để chống trùng."""
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(str(random.random()).encode()).hexdigest()


def has_audio_stream(filepath: str) -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-i", filepath], capture_output=True, text=True, timeout=10)
        return "Audio:" in r.stderr
    except Exception:
        return False


def get_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip() or 10.0)
    except Exception:
        return 10.0

# ==============================================================================
# FETCH ẢNH — 3 tầng, ưu tiên khớp ngữ nghĩa cao
# ==============================================================================
def fetch_image_pexels(query: str, p_key: str, used_urls: set, used_hashes: set, dest: str) -> bool:
    if not p_key or not p_key.strip():
        return False
    headers = {"Authorization": p_key.strip()}
    for page in random.sample(range(1, 4), 3):
        try:
            url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(query)}&per_page=15&page={page}&orientation=landscape"
            r = requests.get(url, headers=headers, timeout=8)
            if not (r.ok and r.json().get("photos")):
                continue
            photos = r.json()["photos"]
            random.shuffle(photos)
            for p in photos:
                u = p["src"].get("large2x") or p["src"].get("large")
                if not u or u in used_urls:
                    continue
                if download_file(u, dest, min_size=40000):
                    ch = image_content_hash(dest)
                    if ch in used_hashes:
                        continue
                    used_urls.add(u)
                    used_hashes.add(ch)
                    return True
        except Exception:
            continue
    return False


def fetch_image_wikimedia(query: str, used_urls: set, used_hashes: set, dest: str) -> bool:
    try:
        url = (f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
               f"&gsrsearch={urllib.parse.quote(query)}&gsrlimit=15"
               f"&prop=imageinfo&iiprop=url|size&format=json")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if not r.ok:
            return False
        pages = r.json().get("query", {}).get("pages", {})
        items = list(pages.values())
        random.shuffle(items)
        for info in items:
            img_info = info.get("imageinfo", [{}])[0]
            u = img_info.get("url")
            if not u or u in used_urls:
                continue
            if not u.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if img_info.get("width", 0) < 800:
                continue
            if download_file(u, dest, min_size=40000):
                ch = image_content_hash(dest)
                if ch in used_hashes:
                    continue
                used_urls.add(u)
                used_hashes.add(ch)
                return True
    except Exception:
        pass
    return False


def fetch_image_ddg(query: str, used_urls: set, used_hashes: set, dest: str, is_english: bool) -> bool:
    try:
        region = "wt-wt" if is_english else "vn-vi"
        with DDGS() as ddgs:
            results = list(ddgs.images(query, region=region, max_results=12))
        random.shuffle(results)
        for r in results:
            u = r.get("image")
            if not u or not u.startswith("http") or u in used_urls:
                continue
            if download_file(u, dest, min_size=45000, timeout=6):
                try:
                    with Image.open(dest) as t:
                        if t.size[0] < 700 or t.size[1] < 400:
                            continue
                        t.verify()
                except Exception:
                    continue
                ch = image_content_hash(dest)
                if ch in used_hashes:
                    continue
                used_urls.add(u)
                used_hashes.add(ch)
                return True
    except Exception:
        pass
    return False


def fetch_matching_image(query_vn: str, query_en: str, idx: int, workdir: str,
                         used_urls: set, used_hashes: set, p_key: str, is_english: bool) -> str:
    """
    Tìm ảnh khớp với query. Thử 3 tầng với nhiều biến thể query.
    """
    dest = os.path.join(workdir, f"img_{idx:03d}.jpg")
    search_en = (query_en or "").strip() or "everyday life scene"
    search_vn = (query_vn or "").strip()

    # Nhiều biến thể query để tăng tỉ lệ khớp
    query_variants = [search_en]
    # Bỏ từ nhiễu và tạo biến thể ngắn hơn
    words = [w for w in re.findall(r'[a-zA-Z]+', search_en) if len(w) > 2]
    if len(words) >= 4:
        query_variants.append(" ".join(words[:4]))
    if len(words) >= 2:
        query_variants.append(" ".join(words[:2]))

    # Tầng 1: Pexels (chất lượng cao, khớp ngữ nghĩa tốt)
    for q in query_variants:
        if fetch_image_pexels(q, p_key, used_urls, used_hashes, dest):
            return _finalize_image(dest)
        if fetch_image_pexels(q + " realistic photo", p_key, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 2: Wikimedia (miễn phí, đáng tin)
    for q in query_variants:
        if fetch_image_wikimedia(q, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 3: DuckDuckGo (nhiều kết quả nhưng cần lọc)
    for q in query_variants + ([search_vn] if search_vn and not is_english else []):
        if fetch_image_ddg(q, used_urls, used_hashes, dest, is_english):
            return _finalize_image(dest)
        if fetch_image_ddg(q + " photo", used_urls, used_hashes, dest, is_english):
            return _finalize_image(dest)

    # Fallback an toàn cuối cùng
    _make_safe_fallback(dest, idx)
    return dest


def _finalize_image(path: str) -> str:
    """Cắt fit về 1280x720, giữ ảnh gốc chân thực (bỏ color grading giả tạo)."""
    try:
        with Image.open(path) as im:
            fitted = ImageOps.fit(im.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(path, "JPEG", quality=92)
    except Exception:
        _make_safe_fallback(path, 0)
    return path


def _make_safe_fallback(path: str, idx: int):
    """Fallback tối thiểu: gradient tối giản (không random ảnh rác)."""
    img = Image.new("RGB", (W, H), (18, 22, 32))
    img.save(path, "JPEG", quality=88)

# ==============================================================================
# FETCH VIDEO B-ROLL (có âm thanh gốc)
# ==============================================================================
def fetch_broll_clip(query_en: str, idx: int, target_frames: int, p_key: str,
                     workdir: str, used_vid_ids: set) -> str:
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    dur = target_frames / FPS
    headers = {"Authorization": p_key.strip()}

    terms = [query_en]
    words = [w for w in re.findall(r'[a-zA-Z]+', query_en) if len(w) > 2]
    if len(words) >= 3:
        terms.append(" ".join(words[:3]))
    terms.append(f"{query_en} cinematic")

    for term in terms:
        for page in random.sample(range(1, 4), 3):
            try:
                url = (f"{PEXELS_VIDEO_URL}?query={urllib.parse.quote(term)}"
                       f"&per_page=10&page={page}&orientation=landscape")
                r = requests.get(url, headers=headers, timeout=8)
                if not (r.ok and r.json().get("videos")):
                    continue
                videos = r.json()["videos"]
                random.shuffle(videos)
                for v in videos:
                    v_id = v.get("id")
                    if not v_id or v_id in used_vid_ids:
                        continue
                    files = v.get("video_files", [])
                    hd = [f for f in files if f.get("height", 0) >= 720 and f.get("file_type") == "video/mp4"]
                    pick = (hd or files)
                    if not pick:
                        continue
                    target_url = pick[0].get("link")
                    if not target_url:
                        continue
                    if download_file(target_url, raw_vid, min_size=80000, timeout=20):
                        used_vid_ids.add(v_id)
                        try:
                            keep_audio = has_audio_stream(raw_vid)
                            vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p"
                            cmd = ["ffmpeg", "-y", "-i", raw_vid, "-t", f"{dur:.3f}", "-vf", vf]
                            if keep_audio:
                                cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                                        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", clip_dest]
                            else:
                                cmd += ["-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", clip_dest]
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            try:
                                os.remove(raw_vid)
                            except OSError:
                                pass
                            return clip_dest
                        except Exception:
                            if os.path.exists(raw_vid):
                                os.remove(raw_vid)
            except Exception:
                continue
    return None

# ==============================================================================
# KEN BURNS CLIP từ ảnh
# ==============================================================================
def create_kenburns_clip(img_path: str, target_frames: int, out_clip: str, mode: int = 0):
    frames = max(25, target_frames)
    dur = frames / FPS
    step = 0.15 / frames
    m = mode % 4

    if not os.path.exists(img_path) or os.path.getsize(img_path) < 3000:
        _make_safe_fallback(img_path, 0)

    if m == 0:
        z, x, y = f"min(zoom+{step:.6f},1.15)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif m == 1:
        z, x, y = "1.15", f"(iw-iw/zoom)*(on/{frames})", "ih/2-(ih/zoom/2)"
    elif m == 2:
        z, x, y = f"if(eq(on,1),1.15,max(1.0,zoom-{step:.6f}))", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    else:
        z, x, y = "1.15", f"(iw-iw/zoom)*(1-on/{frames})", "ih/2-(ih/zoom/2)"

    vf = (f"scale=2560:1440,"
          f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s=2560x1440:fps={FPS},"
          f"scale={W}:{H}:flags=lanczos,format=yuv420p")

    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", img_path, "-vf", vf,
           "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", out_clip]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ==============================================================================
# AI TRÍCH XUẤT QUERY KHỚP VOICE
# ==============================================================================
def ai_extract_queries(client, segments_batch, b_start, is_english):
    """
    Bóc tách hành động-vật thể-bối cảnh CHÍNH XÁC theo từng câu thoại.
    3 tầng bắt buộc: SUBJECT + ACTION + SETTING. Không bias theo genre.
    """
    lines = "\n".join([f"[{i + b_start}] {s['text'][:140]}" for i, s in enumerate(segments_batch)])

    prompt = f"""Bạn là một Visual Director chuyên nghiệp, có nhiệm vụ chuyển từng câu thoại thành mô tả hình ảnh CỤ THỂ để tìm stock footage khớp 100%.

QUY TẮC BẮT BUỘC:
1. Bóc tách theo 3 tầng: [CHỦ THỂ CỤ THỂ] + [HÀNH ĐỘNG VẬT LÝ] + [BỐI CẢNH].
   Ví dụ: "cô gái rót cà phê quán nhỏ", "nam sinh mở cửa lớp học", "xe máy chạy đường mưa".

2. KHÔNG suy diễn cảm xúc trừu tượng. CHỈ mô tả những gì camera có thể quay được.
   CẤM: sad, lonely, depressed, thinking, moody, deep, feeling, vibe.

3. Nếu câu thoại nhắc đến vật thể cụ thể (điện thoại, ly cà phê, quyển sách, xe, áo, v.v.) → PHẢI đưa vật đó vào query.

4. Nếu câu thoại nhắc địa điểm cụ thể (quán ăn, trường học, công viên, văn phòng) → PHẢI đưa địa điểm đó vào query.

5. query_en PHẢI 4–8 từ tiếng Anh, dùng danh từ + động từ cụ thể. KHÔNG dùng từ chung chung như "life", "moment", "scene".

6. Mỗi câu thoại → 1 query RIÊNG, KHÔNG trùng với câu khác.

Đoạn thoại:
{lines}

Trả về DUY NHẤT JSON:
{{"scenes": [
  {{"index": {b_start}, "query_en": "young woman pouring coffee small cafe", "query_vn": "cô gái rót cà phê quán nhỏ"}}
]}}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        content = resp.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0)).get("scenes", [])
        return {int(it.get("index", -1)): it for it in parsed if "index" in it}
    except Exception:
        return {}

# ==============================================================================
# PIPELINE CHÍNH
# ==============================================================================
if st.button("⚡ Bắt Đầu Dựng Video Thành Phẩm", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice lên trước!")
    else:
        status = st.status("Đang khởi động xưởng sản xuất...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="master_prod_")
        used_urls = set()
        used_img_hashes = st.session_state.used_img_hashes
        used_vid_ids = st.session_state.used_vid_ids

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            total_audio_dur = get_duration(audio_path)
            total_required_frames = int(round(total_audio_dur * FPS))
            client = Groq(api_key=groq_key.strip())

            # ---------- BƯỚC 1: WHISPER ----------
            status.update(label="🎙️ 1/4: Whisper bóc tách timestamp + lời thoại...")
            compressed = os.path.join(workdir, "whisper_input.mp3")
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path, "-vn",
                "-ar", "16000", "-ac", "1", "-b:a", "48k", compressed
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with open(compressed, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []
            detected_lang = (data.get("language") or "vietnamese").lower()
            is_english = "en" in detected_lang

            # Gộp segment thành câu ~4.5s trở lên
            segments = []
            cur_text, cur_start = "", 0.0
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
                segments.append({"start": cur_start, "end": total_audio_dur, "text": cur_text})
            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "story scene"})

            # Phân bổ frame chính xác theo timestamp
            accumulated = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i + 1]["start"] - segments[i]["start"]
                    segments[i]["target_frames"] = max(25, int(round(seg_dur * FPS)))
                    accumulated += segments[i]["target_frames"]
                else:
                    segments[i]["target_frames"] = max(25, total_required_frames - accumulated)

            # ---------- BƯỚC 2: AI BÓC TÁCH QUERY ----------
            status.update(label="🧠 2/4: AI bóc tách hành động–vật thể–bối cảnh theo voice...")
            by_idx = {}
            batch_size = 10
            for b_start in range(0, len(segments), batch_size):
                sub = segments[b_start:b_start + batch_size]
                parsed = ai_extract_queries(client, sub, b_start, is_english)
                by_idx.update(parsed)
                status.write(f"✓ Đã phân tích câu {b_start + 1}–{b_start + len(sub)}")

            # ---------- BƯỚC 3: DỰNG CẢNH ----------
            status.update(label="🎬 3/4: Tìm ảnh/video khớp voice + dựng cảnh...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = (sc_data.get("query_vn") or sc["text"][:60]).strip()
                    query_en = (sc_data.get("query_en") or sc["text"][:60]).strip()
                    t_frames = sc["target_frames"]

                    # Ưu tiên video B-roll ở một số slot, xen kẽ ảnh Ken Burns
                    clip_path = None
                    is_video_slot = (idx % 4 == 1) and (idx != len(segments) - 1)
                    if is_video_slot and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, t_frames, pexels_key,
                                                     workdir, used_vid_ids)

                    if not clip_path:
                        img_path = fetch_matching_image(
                            query_vn, query_en, idx, workdir,
                            used_urls, used_img_hashes, pexels_key, is_english
                        )
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")
                    status.write(f"✓ Cảnh {idx + 1}/{len(segments)}: `{query_en[:50]}`")

            # ---------- BƯỚC 4: XUẤT MASTER ----------
            status.update(label="⚡ 4/4: Ghép master + chuẩn hóa audio...")
            out_path = os.path.join(workdir, "output.mp4")

            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{total_audio_dur:.3f}",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                out_path
            ], check=True)

            status.update(label="✅ Video hoàn thành!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Master",
                data=video_bytes,
                file_name=f"master_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
