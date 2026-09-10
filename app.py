# -*- coding: utf-8 -*-
import os
import re
import io
import sys
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

st.set_page_config(page_title="Studio POV Master Engine Pro", page_icon="🎬", layout="centered")

# ==============================================================================
# FFMPEG RESOLVER — chạy trên Streamlit Cloud & local
# ==============================================================================
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

def _resolve_ffprobe():
    for cand in ["ffprobe", "/usr/bin/ffprobe", "/usr/local/bin/ffprobe"]:
        if cand == "ffprobe":
            p = shutil.which(cand)
            if p:
                return p
        elif os.path.exists(cand):
            return cand
    return None

FFPROBE_EXE = _resolve_ffprobe()


def probe_duration(path):
    """Lấy duration — dùng ffprobe nếu có, fallback parse stderr ffmpeg."""
    if FFPROBE_EXE:
        try:
            r = subprocess.run(
                [FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=15
            )
            return float(r.stdout.strip() or 10.0)
        except Exception:
            pass
    try:
        r = subprocess.run([FFMPEG_EXE, "-i", path],
                           capture_output=True, text=True, timeout=15)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mn * 60 + s
    except Exception:
        pass
    return 10.0

W, H = 1280, 720
FPS = 25
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

# Ngưỡng chia cảnh
MIN_SCENE_DUR = 2.0
MAX_SCENE_DUR = 4.0
TARGET_SCENE_DUR = 2.8

# ==============================================================================
# MODEL CHAINS — CHỈ DÙNG CÁC MODEL CÓ TRONG DANH SÁCH CỦA BẠN
# ==============================================================================
# Query hình ảnh — ưu tiên gpt-oss-20b (khớp voice tốt nhất) → các model mạnh
QUERY_MODEL_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3.6-27b",
]

# SFX + mood — task nhẹ, model nhỏ nhanh
SFX_MODEL_CHAIN = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

MOOD_MODEL_CHAIN = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

# Whisper — turbo trước, full sau, cuối cùng faster-whisper local
WHISPER_MODEL_CHAIN = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

# Mix gains
VOICE_GAIN = 1.00
SFX_GAIN = 0.30
BGM_GAIN = 0.14

# ==============================================================================
# SECRETS
# ==============================================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"].strip()
except Exception:
    st.error("❌ Thiếu `GROQ_API_KEY` trong Streamlit Secrets.")
    st.stop()

PEXELS_API_KEY = st.secrets.get("PEXELS_API_KEY", "").strip()

# ==============================================================================
# SESSION CACHE
# ==============================================================================
for key in ("used_img_hashes", "used_vid_ids", "used_sfx_urls"):
    if key not in st.session_state:
        st.session_state[key] = set()

# ==============================================================================
# UI
# ==============================================================================
st.title("🎬 Studio POV Master Engine Pro")
st.caption("Khớp voice 100% • SFX ambient • BGM mood • Ducking chuyên nghiệp")

audio_file = st.file_uploader("Tải lên file Voice (MP3, WAV, M4A, OGG)",
                              type=["mp3", "wav", "m4a", "ogg"])

col_a, col_b = st.columns(2)
with col_a:
    enable_sfx = st.checkbox("🔊 Thêm SFX ambient", value=True)
    enable_bgm = st.checkbox("🎵 Thêm nhạc nền theo mood", value=True)
with col_b:
    enable_motion = st.checkbox("🎞️ Motion cho video B-roll", value=True)
    enable_ducking = st.checkbox("🎚️ Auto-ducking BGM", value=True)

if not PEXELS_API_KEY:
    st.warning("⚠️ Chưa có `PEXELS_API_KEY` — sẽ dùng ảnh Wikimedia/DDG (chất lượng thấp hơn).")

# ==============================================================================
# HELPERS
# ==============================================================================
def log(msg):
    print(f"[POV] {msg}", flush=True)


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


def has_audio_stream(fp):
    try:
        r = subprocess.run([FFMPEG_EXE, "-i", fp],
                           capture_output=True, text=True, timeout=10)
        return "Audio:" in r.stderr
    except Exception:
        return False


# ==============================================================================
# MODEL CALLERS
# ==============================================================================
def call_llm(client, messages, model_chain, temperature=0.2):
    for model in model_chain:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            content = resp.choices[0].message.content
            log(f"LLM OK: {model}")
            return content, model
        except Exception as e:
            log(f"LLM FAIL {model}: {str(e)[:140]}")
            continue
    return None, None


def call_whisper(client, audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    for model in WHISPER_MODEL_CHAIN:
        try:
            resp = client.audio.transcriptions.create(
                file=(audio_path, io.BytesIO(audio_bytes)),
                model=model, response_format="verbose_json"
            )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            log(f"Whisper OK: {model}")
            return data
        except Exception as e:
            log(f"Whisper FAIL {model}: {str(e)[:140]}")
    log("Thử faster-whisper local...")
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel("small", device="cpu", compute_type="int8")
        segs, info = wm.transcribe(audio_path, language=None)
        return {
            "language": info.language,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segs],
        }
    except Exception as e:
        log(f"faster-whisper fail: {e}")
        return None


# ==============================================================================
# CHIA CẢNH
# ==============================================================================
def split_text_by_punctuation(text):
    parts = re.split(
        r'(?<=[.!?;])\s+|(?<=,)\s+(?=[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ])',
        text
    )
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    return parts or [text]


def build_segments_from_whisper(raw_segs, total_dur):
    raw_sentences, buf_text, buf_start = [], "", 0.0
    for seg in raw_segs:
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        if not buf_text:
            buf_start = float(seg["start"])
        buf_text = (buf_text + " " + t).strip()
        if re.search(r'[.!?]\s*$', buf_text):
            raw_sentences.append({"start": buf_start, "end": float(seg["end"]), "text": buf_text})
            buf_text = ""
    if buf_text:
        raw_sentences.append({"start": buf_start, "end": total_dur, "text": buf_text})
    if not raw_sentences:
        raw_sentences = [{"start": 0.0, "end": total_dur, "text": "story scene"}]

    segments = []
    for sent in raw_sentences:
        dur = max(0.1, sent["end"] - sent["start"])
        text = sent["text"]
        if dur <= MAX_SCENE_DUR:
            segments.append({"start": sent["start"], "end": sent["end"], "text": text})
            continue
        parts = split_text_by_punctuation(text)
        if len(parts) <= 1:
            words = text.split()
            n_chunks = max(2, math.ceil(dur / TARGET_SCENE_DUR))
            cs = max(2, len(words) // n_chunks)
            parts = [" ".join(words[i:i + cs]) for i in range(0, len(words), cs)]
        total_chars = sum(len(p) for p in parts) or 1
        cur_t = sent["start"]
        for p in parts:
            ratio = len(p) / total_chars
            part_dur = max(MIN_SCENE_DUR * 0.6, dur * ratio)
            end_t = min(sent["end"], cur_t + part_dur)
            segments.append({"start": cur_t, "end": end_t, "text": p.strip()})
            cur_t = end_t
        if segments and abs(segments[-1]["end"] - sent["end"]) > 0.05:
            segments[-1]["end"] = sent["end"]

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
    if not p_key:
        return False
    headers = {"Authorization": p_key}
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
        r = requests.get(url, headers={"User-Agent": "POVMaster/1.0"}, timeout=8)
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
            results = list(ddgs.images(query, region=region, max_results=8))
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
    except Exception as e:
        log(f"DDG fail: {str(e)[:80]}")
    return False


def _finalize_image(path):
    try:
        with Image.open(path) as im:
            fitted = ImageOps.fit(im.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(path, "JPEG", quality=92)
    except Exception:
        Image.new("RGB", (W, H), (18, 22, 32)).save(path, "JPEG", quality=88)
    return path


def fetch_matching_image(query_candidates, idx, workdir, used_urls, used_hashes, p_key, is_english):
    """Thử lần lượt các candidate + biến thể từ khóa trên 3 tầng Pexels → Wikimedia → DDG."""
    dest = os.path.join(workdir, f"img_{idx:03d}.jpg")

    all_variants = []
    for q in query_candidates:
        q = (q or "").strip()
        if not q:
            continue
        all_variants.append(q)
        words = re.findall(r'[\wàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+',
                           q, re.IGNORECASE)
        if len(words) >= 4:
            all_variants.append(" ".join(words[:4]))
        if len(words) >= 2:
            all_variants.append(" ".join(words[:2]))

    seen, variants = set(), []
    for v in all_variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            variants.append(v)

    # Tầng 1: Pexels
    for q in variants:
        if fetch_image_pexels(q, p_key, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 2: Wikimedia
    for q in variants:
        if fetch_image_wikimedia(q, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 3: DuckDuckGo (chỉ 3 variant đầu, tránh rate-limit)
    for q in variants[:3]:
        if fetch_image_ddg(q, used_urls, used_hashes, dest, is_english):
            return _finalize_image(dest)

    Image.new("RGB", (W, H), (18, 22, 32)).save(dest, "JPEG", quality=88)
    return dest


# ==============================================================================
# FETCH VIDEO B-ROLL
# ==============================================================================
def fetch_broll_clip(query_candidates, idx, target_frames, p_key, workdir, used_vid_ids, add_motion=False):
    if not p_key:
        return None
    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    dur = target_frames / FPS
    headers = {"Authorization": p_key}

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
                    pick = hd or files
                    if not pick:
                        continue
                    target_url = pick[0].get("link")
                    if not target_url:
                        continue
                    if download_file(target_url, raw_vid, min_size=80000, timeout=20):
                        used_vid_ids.add(v_id)
                        try:
                            keep_audio = has_audio_stream(raw_vid)
                            if add_motion:
                                step = 0.03 / max(1, target_frames)
                                vf = (f"scale={int(W*1.1)}:{int(H*1.1)},"
                                      f"zoompan=z='min(zoom+{step:.6f},1.03)':"
                                      f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                                      f"d={target_frames}:s={W}x{H}:fps={FPS},format=yuv420p")
                            else:
                                vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                                      f"crop={W}:{H},fps={FPS},format=yuv420p")
                            cmd = [FFMPEG_EXE, "-y", "-i", raw_vid, "-t", f"{dur:.3f}", "-vf", vf]
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
        Image.new("RGB", (W, H), (18, 22, 32)).save(img_path, "JPEG", quality=88)
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
        FFMPEG_EXE, "-y", "-loop", "1", "-i", img_path, "-vf", vf,
        "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", out_clip
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ==============================================================================
# AI — TÁCH 2 TASK: QUERY HÌNH ẢNH (PROMPT FEW-SHOT BẢN CŨ) + SFX
# ==============================================================================
def ai_extract_queries(client, seg_batch, b_start, is_english):
    """
    Trích xuất 3 query tiếng Anh + 1 query VN cho mỗi câu thoại.
    PROMPT GIỮ NGUYÊN few-shot examples của bản cũ để khớp voice.
    """
    lines = "\n".join([f"[{i + b_start}] {s['text']}" for i, s in enumerate(seg_batch)])

    prompt = f"""Bạn là một đạo diễn hình ảnh có khả năng thích ứng linh hoạt tuyệt đối.
Nhiệm vụ: Lắng nghe từng câu thoại tiếng {"Anh" if is_english else "Việt"} và trích xuất ĐÚNG HÀNH ĐỘNG, ĐỊA ĐIỂM, VẬT THỂ được nói đến.

QUY TẮC CỐT LÕI:

1. KHÔNG ÉP BẤT KỲ ĐỊNH KIẾN NÀO — BÁM SÁT TỪNG TỪ TRONG CÂU GỐC:
   - "đi uống cà phê, nhân viên nữ nhìn trộm" → 'coffee shop barista girl smiling customer table'
   - "hai người bạn đi cạnh nhau, so sánh ngoại hình" → 'two young men walking street outdoor candid'
   - "xin chụp ảnh, đưa điện thoại" → 'people taking selfie photo smartphone smiling outdoor'
   - "xe máy, đường mưa" → 'motorcycle road heavy rain'
   - "ngồi trong lớp học nhìn ra cửa sổ" → 'student sitting classroom window looking outside'
   - "mẹ nấu ăn trong bếp" → 'mother cooking kitchen home warm'
   - "đêm khuya, thành phố sáng đèn" → 'city street night neon lights traffic'
   - "bãi biển, sóng vỗ" → 'beach ocean waves sunset'
   - "đứa trẻ khóc giữa đám đông" → 'crying child crowded market square'
   - "nhân viên văn phòng gõ bàn phím" → 'office worker typing laptop keyboard desk'

2. BẮT BUỘC dùng cấu trúc: [ĐỐI TƯỢNG CỤ THỂ] + [HÀNH ĐỘNG THỰC TẾ] + [ĐỊA ĐIỂM / BỐI CẢNH].

3. TUYỆT ĐỐI CẤM dùng các từ cảm xúc mơ hồ: 'sad', 'depressed', 'lonely', 'thinking', 'suffering', 'moody', 'vibe', 'feeling', 'emotional'.

4. CẤM các từ chung chung: 'life', 'moment', 'scene', 'person', 'thing', 'concept', 'abstract'.

5. Nếu câu thoại nhắc đến VẬT THỂ cụ thể (điện thoại, ly cà phê, quyển sách, xe, áo, cửa, bàn, ghế...) → PHẢI đưa vật đó vào query tiếng Anh.

6. Nếu câu thoại nhắc đến ĐỊA ĐIỂM cụ thể (quán ăn, trường học, công viên, văn phòng, biển, chợ...) → PHẢI đưa địa điểm đó vào query.

7. Trả về 3 query tiếng Anh KHÁC NHAU:
   - q1: CHÍNH XÁC NHẤT — bám sát từng danh từ + động từ của câu gốc
   - q2: biến thể dùng từ đồng nghĩa hoặc góc máy khác (close-up / wide / action)
   - q3: mở rộng vẫn cùng chủ đề, dùng khi q1 và q2 fail

8. Mỗi query 4–8 từ tiếng Anh, viết như search query thực tế (không viết câu hoàn chỉnh).

9. query_vn: bản dịch tiếng Việt ngắn 4–8 từ (giữ NGUYÊN danh từ riêng + động từ của câu gốc để backup search).

Đoạn thoại:
{lines}

Trả về DUY NHẤT JSON:
{{"scenes": [
  {{"index": {b_start}, "queries": ["q1", "q2", "q3"], "query_vn": "cụm tiếng Việt"}}
]}}"""

    content, model = call_llm(
        client,
        messages=[
            {"role": "system", "content": "Bạn CHỈ trả về JSON hợp lệ, không markdown, không giải thích."},
            {"role": "user", "content": prompt},
        ],
        model_chain=QUERY_MODEL_CHAIN,
        temperature=0.25,
    )
    if not content:
        return {}
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0)).get("scenes", [])
    except Exception:
        return {}
    result = {}
    for it in parsed:
        if "index" not in it:
            continue
        idx = int(it["index"])
        qs = it.get("queries") or []
        if isinstance(qs, str):
            qs = [qs]
        result[idx] = {
            "queries": [q for q in qs if q],
            "query_vn": (it.get("query_vn") or "").strip(),
        }
    log(f"Queries extracted by {model}: {len(result)} scenes")
    return result


def ai_extract_sfx(client, seg_batch, b_start):
    """Trích xuất SFX keyword — task nhẹ, dùng model nhỏ."""
    lines = "\n".join([f"[{i + b_start}] {s['text']}" for i, s in enumerate(seg_batch)])

    prompt = f"""Với MỖI câu thoại dưới đây, trả về 1 cụm 2-3 từ tiếng Anh mô tả âm thanh nền phù hợp nhất.

Danh sách gợi ý: rain heavy, rain light, birds chirping, footsteps street, traffic city,
ocean waves, wind trees, engine car, motorcycle, cafe ambience, crowd chatter,
typing keyboard, waterfall, thunder storm, night crickets, church bells, room tone.

Quy tắc:
- Nếu câu nói về hành động ở ngoài trời → chọn âm thanh ngoài trời.
- Nếu câu nói về cảm xúc nội tâm / suy nghĩ → "room tone".
- Nếu không có âm thanh đặc trưng → "room tone".

Đoạn thoại:
{lines}

Trả về DUY NHẤT JSON:
{{"sfx": [{{"index": {b_start}, "sound": "rain heavy"}}]}}"""

    content, _ = call_llm(
        client,
        messages=[
            {"role": "system", "content": "CHỈ trả về JSON."},
            {"role": "user", "content": prompt},
        ],
        model_chain=SFX_MODEL_CHAIN,
        temperature=0.1,
    )
    if not content:
        return {}
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0)).get("sfx", [])
    except Exception:
        return {}
    return {int(it["index"]): (it.get("sound") or "").strip()
            for it in parsed if "index" in it}


def ai_pick_bgm_mood(client, full_transcript):
    prompt = f"""Đọc transcript sau và chọn 1 mood nhạc nền phù hợp.

Mood: epic, sad, happy, chill, tense, mystery, motivational, romantic.

Chỉ trả về 1 từ tiếng Anh (không markdown, không giải thích).

Transcript:
{full_transcript[:2000]}

Mood:"""
    content, _ = call_llm(
        client,
        messages=[{"role": "user", "content": prompt}],
        model_chain=MOOD_MODEL_CHAIN,
        temperature=0.1,
    )
    if not content:
        return "chill"
    mood = content.strip().lower().split()[0]
    valid = {"epic", "sad", "happy", "chill", "tense", "mystery", "motivational", "romantic"}
    return mood if mood in valid else "chill"


# ==============================================================================
# SFX + BGM FETCHERS
# ==============================================================================
def fetch_sfx_myinstants(sfx_query, dest, used_sfx_urls):
    try:
        search_url = f"https://www.myinstants.com/en/search/?name={urllib.parse.quote(sfx_query)}"
        r = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if not r.ok:
            return False
        mp3s = re.findall(r'(/media/sounds/[^"\']+\.mp3)', r.text)
        random.shuffle(mp3s)
        for rel in mp3s[:5]:
            full = "https://www.myinstants.com" + rel
            if full in used_sfx_urls:
                continue
            if download_file(full, dest, min_size=2000, timeout=6):
                used_sfx_urls.add(full)
                return True
    except Exception:
        pass
    return False


def fetch_sfx_wikimedia(sfx_query, dest, used_sfx_urls):
    try:
        api = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
               f"&gsrsearch={urllib.parse.quote(sfx_query + ' filetype:audio')}&gsrlimit=10"
               "&prop=imageinfo&iiprop=url|size&format=json")
        r = requests.get(api, headers={"User-Agent": "POVMaster/1.0"}, timeout=8)
        if not r.ok:
            return False
        items = list(r.json().get("query", {}).get("pages", {}).values())
        random.shuffle(items)
        for info in items:
            u = info.get("imageinfo", [{}])[0].get("url")
            if not u or u in used_sfx_urls:
                continue
            if not u.lower().endswith((".mp3", ".ogg", ".wav", ".flac")):
                continue
            if download_file(u, dest, min_size=2000, timeout=8):
                used_sfx_urls.add(u)
                return True
    except Exception:
        pass
    return False


def fetch_sfx(sfx_query, idx, workdir, used_sfx_urls):
    if not sfx_query or sfx_query == "room tone":
        return None
    dest = os.path.join(workdir, f"sfx_{idx:03d}.mp3")
    if fetch_sfx_myinstants(sfx_query, dest, used_sfx_urls):
        return dest
    if fetch_sfx_wikimedia(sfx_query, dest, used_sfx_urls):
        return dest
    return None


BGM_LIBRARY = {
    "epic":         ["https://upload.wikimedia.org/wikipedia/commons/4/4c/Scott_Buckley_-_Aurora.mp3"],
    "sad":          ["https://upload.wikimedia.org/wikipedia/commons/2/2b/Kai_Engel_-_03_-_Great_Expectations.ogg"],
    "happy":        ["https://upload.wikimedia.org/wikipedia/commons/f/f1/Kai_Engel_-_08_-_Soft.ogg"],
    "chill":        ["https://upload.wikimedia.org/wikipedia/commons/4/4c/Scott_Buckley_-_Aurora.mp3"],
    "tense":        ["https://upload.wikimedia.org/wikipedia/commons/3/30/Kai_Engel_-_01_-_Growing_Up.ogg"],
    "mystery":      ["https://upload.wikimedia.org/wikipedia/commons/2/2b/Kai_Engel_-_03_-_Great_Expectations.ogg"],
    "motivational": ["https://upload.wikimedia.org/wikipedia/commons/4/4c/Scott_Buckley_-_Aurora.mp3"],
    "romantic":     ["https://upload.wikimedia.org/wikipedia/commons/f/f1/Kai_Engel_-_08_-_Soft.ogg"],
}


def fetch_bgm(mood, workdir):
    urls = BGM_LIBRARY.get(mood, BGM_LIBRARY["chill"])
    dest = os.path.join(workdir, "bgm.mp3")
    for url in urls:
        if download_file(url, dest, min_size=50000, timeout=20):
            return dest
    return None


# ==============================================================================
# AUDIO MIXER
# ==============================================================================
def build_audio_track(voice_path, sfx_entries, bgm_path, total_dur, out_audio):
    """Voice 100% + SFX 30% (chèn tại timestamp) + BGM 14% (loop) → amix → loudnorm."""
    inputs = ["-i", voice_path]
    filter_parts = [f"[0:a]volume={VOICE_GAIN},apad=pad_dur={total_dur},atrim=0:{total_dur}[v]"]
    mix_streams = ["[v]"]
    idx = 1

    for sfx_path, s_start, s_dur in sfx_entries:
        if not sfx_path or not os.path.exists(sfx_path):
            continue
        inputs += ["-i", sfx_path]
        delay_ms = max(0, int(s_start * 1000))
        fade_out = max(0.1, s_dur - 0.5)
        filter_parts.append(
            f"[{idx}:a]volume={SFX_GAIN},atrim=0:{s_dur},"
            f"afade=t=in:d=0.4,afade=t=out:st={fade_out:.2f}:d=0.5,"
            f"adelay={delay_ms}|{delay_ms},apad=pad_dur={total_dur},"
            f"atrim=0:{total_dur}[sfx{idx}]"
        )
        mix_streams.append(f"[sfx{idx}]")
        idx += 1

    if bgm_path and os.path.exists(bgm_path):
        inputs += ["-stream_loop", "-1", "-i", bgm_path]
        filter_parts.append(
            f"[{idx}:a]volume={BGM_GAIN},atrim=0:{total_dur},"
            f"afade=t=in:d=1.5,afade=t=out:st={max(0, total_dur - 2):.2f}:d=2[bgm]"
        )
        mix_streams.append("[bgm]")
        idx += 1

    filter_parts.append(
        f"{''.join(mix_streams)}amix=inputs={len(mix_streams)}:duration=first:"
        f"dropout_transition=2,loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    )

    cmd = [FFMPEG_EXE, "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", out_audio,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


# ==============================================================================
# PIPELINE CHÍNH
# ==============================================================================
if st.button("⚡ Bắt Đầu Dựng Video Thành Phẩm", use_container_width=True, type="primary"):
    if not audio_file:
        st.error("Vui lòng tải file Voice lên trước!")
    else:
        status = st.status("Đang khởi động xưởng sản xuất...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="master_prod_")
        used_urls = set()
        used_img_hashes = st.session_state.used_img_hashes
        used_vid_ids = st.session_state.used_vid_ids
        used_sfx_urls = st.session_state.used_sfx_urls

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            total_audio_dur = probe_duration(audio_path)
            total_required_frames = int(round(total_audio_dur * FPS))
            client = Groq(api_key=GROQ_API_KEY)
            log(f"Audio duration: {total_audio_dur:.2f}s")

            # ---------- 1. WHISPER ----------
            status.update(label="🎙️ 1/5: Whisper bóc tách timestamp...")
            compressed = os.path.join(workdir, "whisper_input.mp3")
            subprocess.run([
                FFMPEG_EXE, "-y", "-i", audio_path, "-vn",
                "-ar", "16000", "-ac", "1", "-b:a", "48k", compressed
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            data = call_whisper(client, compressed)
            if not data:
                st.error("❌ Không transcribe được. Bật Whisper trong Groq console "
                         "(tab Audio) hoặc thêm `faster-whisper` vào requirements.txt.")
                st.stop()

            raw_segs = data.get("segments") or []
            detected_lang = (data.get("language") or "vietnamese").lower()
            is_english = "en" in detected_lang
            log(f"Detected language: {detected_lang}")

            # ---------- 2. CHIA CẢNH ----------
            segments = build_segments_from_whisper(raw_segs, total_audio_dur)
            status.write(f"📊 Chia thành **{len(segments)} cảnh** "
                         f"(~{total_audio_dur / len(segments):.1f}s/cảnh)")

            accumulated = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i + 1]["start"] - segments[i]["start"]
                    segments[i]["target_frames"] = max(15, int(round(seg_dur * FPS)))
                    accumulated += segments[i]["target_frames"]
                else:
                    segments[i]["target_frames"] = max(15, total_required_frames - accumulated)

            # ---------- 3. AI — QUERY HÌNH ẢNH (task chính, prompt few-shot) ----------
            status.update(label="🧠 2/5: AI bóc tách query hình ảnh theo từng câu thoại...")
            by_idx = {}
            batch_size = 8
            for b_start in range(0, len(segments), batch_size):
                sub = segments[b_start:b_start + batch_size]
                by_idx.update(ai_extract_queries(client, sub, b_start, is_english))
                status.write(f"✓ Query: câu {b_start + 1}–{b_start + len(sub)}")

            # ---------- 4. AI — SFX (task phụ, model nhẹ, tách riêng) ----------
            sfx_by_idx = {}
            if enable_sfx:
                status.update(label="🔊 2.5/5: AI chọn SFX ambient cho từng cảnh...")
                for b_start in range(0, len(segments), batch_size):
                    sub = segments[b_start:b_start + batch_size]
                    sfx_by_idx.update(ai_extract_sfx(client, sub, b_start))
                non_tone = sum(1 for v in sfx_by_idx.values() if v and v != "room tone")
                status.write(f"✓ SFX: {non_tone}/{len(segments)} cảnh có âm thanh riêng")

            # ---------- 5. MOOD NHẠC + TẢI BGM ----------
            full_transcript = " ".join([s["text"] for s in segments])
            bgm_mood = ai_pick_bgm_mood(client, full_transcript) if enable_bgm else "chill"
            status.write(f"🎵 Mood nhạc nền: **{bgm_mood}**")

            bgm_path = None
            if enable_bgm:
                bgm_path = fetch_bgm(bgm_mood, workdir)
                if bgm_path:
                    status.write("✓ Đã tải nhạc nền")

            # ---------- 6. DỰNG CẢNH + TẢI SFX ----------
            status.update(label="🎬 3/5: Tìm ảnh/video khớp voice + tải SFX...")
            clips_txt = os.path.join(workdir, "clips.txt")
            sfx_entries = []
            progress = st.progress(0.0)

            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    queries = sc_data.get("queries") or []
                    query_vn = sc_data.get("query_vn") or ""

                    # Fallback nếu AI fail
                    if not queries:
                        if is_english:
                            words = [w for w in re.findall(r'[a-zA-Z]+', sc["text"]) if len(w) > 3]
                            queries = [" ".join(words[:5])] if words else ["everyday scene"]
                        else:
                            clean = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
                                           '', sc["text"], flags=re.IGNORECASE).strip()
                            queries = [clean[:80] if clean else "everyday scene"]

                    candidates = list(queries)
                    if query_vn and query_vn not in candidates:
                        candidates.append(query_vn)

                    t_frames = sc["target_frames"]
                    clip_path = None
                    is_video_slot = (idx % 2 == 1) and (idx != len(segments) - 1)

                    if is_video_slot and PEXELS_API_KEY:
                        clip_path = fetch_broll_clip(
                            candidates, idx, t_frames, PEXELS_API_KEY,
                            workdir, used_vid_ids, add_motion=enable_motion
                        )

                    if not clip_path:
                        img_path = fetch_matching_image(
                            candidates, idx, workdir, used_urls,
                            used_img_hashes, PEXELS_API_KEY, is_english
                        )
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

                    # SFX của cảnh này
                    if enable_sfx:
                        sfx_kw = sfx_by_idx.get(idx, "")
                        if sfx_kw and sfx_kw != "room tone":
                            sfx_p = fetch_sfx(sfx_kw, idx, workdir, used_sfx_urls)
                            if sfx_p:
                                sfx_entries.append((sfx_p, sc["start"], sc["end"] - sc["start"]))

                    tag = "VIDEO" if (is_video_slot and clip_path and os.path.exists(clip_path)) else "IMG"
                    first_q = candidates[0] if candidates else ""
                    status.write(f"✓ [{tag}] Cảnh {idx + 1}/{len(segments)}: `{first_q[:55]}`")
                    progress.progress((idx + 1) / len(segments))

            # ---------- 7. TRỘN AUDIO ----------
            status.update(label="⚡ 4/5: Trộn voice + SFX + BGM...")
            mixed_audio = os.path.join(workdir, "mixed_audio.m4a")
            build_audio_track(
                voice_path=audio_path,
                sfx_entries=sfx_entries,
                bgm_path=bgm_path,
                total_dur=total_audio_dur,
                out_audio=mixed_audio,
            )
            status.write(f"✓ Audio mix: {len(sfx_entries)} SFX + BGM {bgm_mood}")

            # ---------- 8. XUẤT MASTER ----------
            status.update(label="⚡ 5/5: Ghép video master...")
            out_path = os.path.join(workdir, "output.mp4")
            subprocess.run([
                FFMPEG_EXE, "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", mixed_audio,
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{total_audio_dur:.3f}",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                out_path
            ], check=True)

            status.update(
                label=f"✅ Hoàn thành! {len(segments)} cảnh • {len(sfx_entries)} SFX • mood {bgm_mood}",
                state="complete"
            )

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
            import traceback
            st.code(traceback.format_exc())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
