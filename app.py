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

# Ngưỡng chia cảnh MỚI — nhiều cảnh hơn
MIN_SCENE_DUR = 2.0    # tối thiểu 2s/cảnh
MAX_SCENE_DUR = 4.0    # tối đa 4s/cảnh trước khi chẻ
TARGET_SCENE_DUR = 2.8 # mục tiêu ~2.8s/cảnh

if "used_img_hashes" not in st.session_state:
    st.session_state.used_img_hashes = set()
if "used_vid_ids" not in st.session_state:
    st.session_state.used_vid_ids = set()

st.title("🎬 Studio POV Master Engine Pro")
st.caption("Cảnh dày hơn • Query đa tầng • Khớp voice 100%")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Khuyến nghị)", type="password")
audio_file = st.file_uploader("Tải lên file Voice (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

# ==============================================================================
# HÀM PHỤ TRỢ
# ==============================================================================
def download_file(url, dest, headers=None, min_size=30000, timeout=8):
    try:
        r = requests.get(url, headers=headers or {"User-Agent": "Mozilla/5.0"},
                         timeout=timeout, stream=True)
        if r.status_code != 200:
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)
        return os.path.getsize(dest) >= min_size
    except Exception:
        return False


def image_content_hash(path):
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()
    except Exception:
        return hashlib.md5(str(random.random()).encode()).hexdigest()


def has_audio_stream(filepath):
    try:
        r = subprocess.run(["ffmpeg", "-i", filepath], capture_output=True, text=True, timeout=10)
        return "Audio:" in r.stderr
    except Exception:
        return False


def get_duration(path):
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
# CHIA CẢNH DÀY — CHIA THEO DẤU CÂU + CẮT NHỎ CÂU DÀI
# ==============================================================================
def split_text_by_punctuation(text):
    """Chia câu thành các mệnh đề theo dấu câu: . ! ? , ; —"""
    parts = re.split(r'(?<=[.!?;])\s+|(?<=,)\s+(?=[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])', text)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    return parts or [text]


def build_segments_from_whisper(raw_segs, total_dur):
    """
    Xây dựng segment dày:
    1. Gộp whisper segments nhỏ thành câu hoàn chỉnh (theo dấu câu)
    2. Nếu câu > MAX_SCENE_DUR → chẻ tiếp
    """
    # Bước 1: gộp whisper segments thành "câu" theo dấu câu cuối
    raw_sentences = []
    buf_text = ""
    buf_start = 0.0
    for seg in raw_segs:
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        if not buf_text:
            buf_start = float(seg["start"])
        buf_text = (buf_text + " " + t).strip()
        # Nếu kết thúc bằng dấu câu mạnh → chốt câu
        if re.search(r'[.!?]\s*$', buf_text):
            raw_sentences.append({
                "start": buf_start,
                "end": float(seg["end"]),
                "text": buf_text
            })
            buf_text = ""
    if buf_text:
        raw_sentences.append({
            "start": buf_start,
            "end": total_dur,
            "text": buf_text
        })
    if not raw_sentences:
        raw_sentences = [{"start": 0.0, "end": total_dur, "text": "story scene"}]

    # Bước 2: chẻ câu dài thành nhiều cảnh ngắn
    segments = []
    for sent in raw_sentences:
        dur = max(0.1, sent["end"] - sent["start"])
        text = sent["text"]

        # Nếu câu đã ngắn (< MAX) → giữ nguyên
        if dur <= MAX_SCENE_DUR:
            segments.append({"start": sent["start"], "end": sent["end"], "text": text})
            continue

        # Câu dài → chẻ theo dấu câu
        parts = split_text_by_punctuation(text)
        if len(parts) <= 1:
            # Không có dấu câu phụ → chẻ theo số từ
            words = text.split()
            n_chunks = max(2, math.ceil(dur / TARGET_SCENE_DUR))
            chunk_size = max(2, len(words) // n_chunks)
            parts = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        # Phân bổ thời gian theo tỉ lệ độ dài ký tự
        total_chars = sum(len(p) for p in parts) or 1
        cur_t = sent["start"]
        for p in parts:
            ratio = len(p) / total_chars
            part_dur = max(MIN_SCENE_DUR * 0.6, dur * ratio)
            end_t = min(sent["end"], cur_t + part_dur)
            segments.append({"start": cur_t, "end": end_t, "text": p.strip()})
            cur_t = end_t
        # Sửa lệch cuối
        if segments and abs(segments[-1]["end"] - sent["end"]) > 0.05:
            segments[-1]["end"] = sent["end"]

    # Bước 3: gộp các mảnh quá ngắn (< MIN_SCENE_DUR) vào mảnh trước
    merged = []
    for s in segments:
        if merged and (s["end"] - s["start"]) < MIN_SCENE_DUR * 0.6:
            merged[-1]["text"] += " " + s["text"]
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)

    return merged or [{"start": 0.0, "end": total_dur, "text": "story scene"}]

# ==============================================================================
# FETCH ẢNH
# ==============================================================================
def fetch_image_pexels(query, p_key, used_urls, used_hashes, dest):
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


def fetch_image_wikimedia(query, used_urls, used_hashes, dest):
    try:
        url = (f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
               f"&gsrsearch={urllib.parse.quote(query)}&gsrlimit=15"
               f"&prop=imageinfo&iiprop=url|size&format=json")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if not r.ok:
            return False
        items = list(r.json().get("query", {}).get("pages", {}).values())
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


def fetch_image_ddg(query, used_urls, used_hashes, dest, is_english):
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


def _finalize_image(path):
    try:
        with Image.open(path) as im:
            fitted = ImageOps.fit(im.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(path, "JPEG", quality=92)
    except Exception:
        _make_safe_fallback(path)
    return path


def _make_safe_fallback(path):
    img = Image.new("RGB", (W, H), (18, 22, 32))
    img.save(path, "JPEG", quality=88)


def fetch_matching_image(query_candidates, idx, workdir, used_urls, used_hashes, p_key, is_english):
    """
    query_candidates: list các query string (ưu tiên) — thử lần lượt.
    """
    dest = os.path.join(workdir, f"img_{idx:03d}.jpg")

    # Mở rộng biến thể từ mỗi candidate
    all_variants = []
    for q in query_candidates:
        q = (q or "").strip()
        if not q:
            continue
        all_variants.append(q)
        words = [w for w in re.findall(r'[a-zA-Z0-9àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+',
                                       q, re.IGNORECASE)]
        if len(words) >= 4:
            all_variants.append(" ".join(words[:4]))
        if len(words) >= 2:
            all_variants.append(" ".join(words[:2]))

    # Loại trùng
    seen = set()
    variants = []
    for v in all_variants:
        vl = v.lower()
        if vl not in seen:
            seen.add(vl)
            variants.append(v)

    # Tầng 1: Pexels
    for q in variants:
        if fetch_image_pexels(q, p_key, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 2: Wikimedia
    for q in variants:
        if fetch_image_wikimedia(q, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 3: DuckDuckGo
    for q in variants:
        if fetch_image_ddg(q, used_urls, used_hashes, dest, is_english):
            return _finalize_image(dest)

    _make_safe_fallback(dest)
    return dest

# ==============================================================================
# FETCH VIDEO B-ROLL
# ==============================================================================
def fetch_broll_clip(query_candidates, idx, target_frames, p_key, workdir, used_vid_ids):
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    dur = target_frames / FPS
    headers = {"Authorization": p_key.strip()}

    terms = []
    for q in query_candidates:
        q = (q or "").strip()
        if q:
            terms.append(q)
            words = [w for w in re.findall(r'[a-zA-Z]+', q) if len(w) > 2]
            if len(words) >= 3:
                terms.append(" ".join(words[:3]))

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
# KEN BURNS
# ==============================================================================
def create_kenburns_clip(img_path, target_frames, out_clip, mode=0):
    frames = max(25, target_frames)
    dur = frames / FPS
    step = 0.15 / frames
    m = mode % 4

    if not os.path.exists(img_path) or os.path.getsize(img_path) < 3000:
        _make_safe_fallback(img_path)

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

    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-vf", vf,
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", out_clip
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ==============================================================================
# AI BÓC TÁCH — TRẢ 3 QUERY CANDIDATES / CẢNH
# ==============================================================================
def ai_extract_query_candidates(client, seg_batch, b_start, is_english):
    """
    Mỗi cảnh → 3 query candidates (EN ưu tiên, VN làm dự phòng).
    """
    lines = "\n".join([f"[{i + b_start}] {s['text']}" for i, s in enumerate(seg_batch)])

    prompt = f"""Bạn là Visual Director chuyên tìm stock footage khớp 100% với lời thoại tiếng {"Anh" if is_english else "Việt"}.

Với MỖI câu thoại, trả về 3 query tiếng Anh KHÁC NHAU để tìm hình ảnh/video khớp nhất:
- query_1 (chính xác nhất): [CHỦ THỂ] + [HÀNH ĐỘNG] + [BỐI CẢNH cụ thể]
- query_2 (gần nghĩa): biến thể dùng từ đồng nghĩa hoặc góc máy khác
- query_3 (dự phòng): mở rộng vẫn cùng chủ đề

QUY TẮC BẮT BUỘC:
1. Phải bám vào DANH TỪ và ĐỘNG TỪ có trong câu thoại gốc (người, vật, hành động, địa điểm).
2. CẤM từ cảm xúc trừu tượng: sad, lonely, depressed, thinking, moody, vibe, feeling.
3. CẤM từ chung chung: life, moment, scene, person, thing.
4. Mỗi query 4–8 từ tiếng Anh, có thể search được trên Pexels/Google.
5. query_vn: bản dịch tiếng Việt ngắn gọn để backup search.

Đoạn thoại:
{lines}

Trả về DUY NHẤT JSON:
{{"scenes": [
  {{"index": {b_start},
    "queries": ["specific query 1", "synonym query 2", "broader query 3"],
    "query_vn": "cụm từ tiếng Việt ngắn"}}
]}}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group(0)).get("scenes", [])
        result = {}
        for it in parsed:
            if "index" not in it:
                continue
            idx = int(it["index"])
            queries = it.get("queries") or []
            if isinstance(queries, str):
                queries = [queries]
            vn = (it.get("query_vn") or "").strip()
            result[idx] = {"queries": [q for q in queries if q], "query_vn": vn}
        return result
    except Exception:
        return {}


def fallback_query_from_text(text, is_english):
    """Nếu AI fail → trích từ khóa trực tiếp từ câu thoại."""
    if is_english:
        words = [w for w in re.findall(r'[a-zA-Z]+', text) if len(w) > 3]
        return [" ".join(words[:5]) if words else "everyday scene"]
    else:
        # Tiếng Việt: dùng nguyên câu làm query VN
        clean = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
                       '', text, flags=re.IGNORECASE).strip()
        return [clean[:80] if clean else "everyday scene"]

# ==============================================================================
# PIPELINE
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

            # ---------- 1. WHISPER ----------
            status.update(label="🎙️ 1/4: Whisper bóc tách timestamp...")
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

            # ---------- 2. CHIA CẢNH DÀY ----------
            segments = build_segments_from_whisper(raw_segs, total_audio_dur)
            status.write(f"📊 Chia thành **{len(segments)} cảnh** (~{total_audio_dur / len(segments):.1f}s/cảnh)")

            # Phân bổ frame chính xác
            accumulated = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i + 1]["start"] - segments[i]["start"]
                    segments[i]["target_frames"] = max(15, int(round(seg_dur * FPS)))
                    accumulated += segments[i]["target_frames"]
                else:
                    segments[i]["target_frames"] = max(15, total_required_frames - accumulated)

            # ---------- 3. AI BÓC TÁCH 3 QUERY/CẢNH ----------
            status.update(label="🧠 3/5: AI sinh 3 query candidates cho mỗi cảnh...")
            by_idx = {}
            batch_size = 8
            for b_start in range(0, len(segments), batch_size):
                sub = segments[b_start:b_start + batch_size]
                parsed = ai_extract_query_candidates(client, sub, b_start, is_english)
                by_idx.update(parsed)
                status.write(f"✓ Đã phân tích câu {b_start + 1}–{b_start + len(sub)}")

            # ---------- 4. DỰNG CẢNH ----------
            status.update(label="🎬 4/5: Tìm ảnh/video khớp voice...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    queries = sc_data.get("queries") or []
                    query_vn = sc_data.get("query_vn") or ""

                    # Nếu AI fail → fallback từ chính câu thoại
                    if not queries:
                        queries = fallback_query_from_text(sc["text"], is_english)
                    # Thêm query_vn làm candidate cuối
                    candidates = list(queries)
                    if query_vn and query_vn not in candidates:
                        candidates.append(query_vn)

                    t_frames = sc["target_frames"]

                    # Tăng tỉ lệ video B-roll lên 1/2 scene
                    clip_path = None
                    is_video_slot = (idx % 2 == 1) and (idx != len(segments) - 1)
                    if is_video_slot and pexels_key:
                        clip_path = fetch_broll_clip(candidates, idx, t_frames,
                                                     pexels_key, workdir, used_vid_ids)

                    if not clip_path:
                        img_path = fetch_matching_image(
                            candidates, idx, workdir,
                            used_urls, used_img_hashes, pexels_key, is_english
                        )
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")
                    tag = "VIDEO" if is_video_slot and clip_path else "IMG"
                    status.write(f"✓ [{tag}] Cảnh {idx + 1}/{len(segments)}: `{(candidates[0] if candidates else '')[:55]}`")

            # ---------- 5. XUẤT MASTER ----------
            status.update(label="⚡ 5/5: Ghép master + loudnorm...")
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

            status.update(label=f"✅ Video hoàn thành ({len(segments)} cảnh)!", state="complete")

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
