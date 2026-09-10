# -*- coding: utf-8 -*-
import os
import re
import io
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
# FFMPEG cho Cloud
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
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 10.0

W, H = 1280, 720
FPS = 25
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

MIN_SCENE_DUR = 2.0
MAX_SCENE_DUR = 4.0
TARGET_SCENE_DUR = 2.8

# ==============================================================================
# MODEL — ĐÚNG như bản cũ
# ==============================================================================
LLM_MODEL = "openai/gpt-oss-20b"
WHISPER_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]

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
for k in ("used_img_hashes", "used_vid_ids"):
    if k not in st.session_state:
        st.session_state[k] = set()

# ==============================================================================
# UI
# ==============================================================================
st.title("🎬 Studio POV Master Engine Pro")
st.caption("Bản cũ — prompt few-shot đầy đủ, khớp voice cao")

audio_file = st.file_uploader("Tải lên file Voice (MP3, WAV, M4A, OGG)",
                              type=["mp3", "wav", "m4a", "ogg"])

with st.expander("🔧 Tùy chọn nâng cao (bỏ trống = chạy y bản cũ)"):
    enable_bgm = st.checkbox("🎵 Thêm nhạc nền loop (mood tĩnh)", value=False)
    enable_motion = st.checkbox("🎞️ Motion nhẹ cho video B-roll (zoom 1.03)", value=False)
    bgm_choice = st.selectbox("Mood nhạc (nếu bật):", ["chill", "epic", "sad", "happy"])

if not PEXELS_API_KEY:
    st.warning("⚠️ Chưa có `PEXELS_API_KEY` — sẽ dùng ảnh Wikimedia/DDG.")

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


def call_whisper(client, audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    for model in WHISPER_MODELS:
        try:
            resp = client.audio.transcriptions.create(
                file=(audio_path, io.BytesIO(audio_bytes)),
                model=model, response_format="verbose_json"
            )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            log(f"Whisper OK: {model}")
            return data
        except Exception as e:
            log(f"Whisper FAIL {model}: {str(e)[:120]}")
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


def fetch_matching_image(query_vn, query_en, idx, workdir, used_urls, used_hashes, p_key, is_english):
    """Bám sát logic bản cũ: query_en chính → biến thể → query_vn."""
    dest = os.path.join(workdir, f"img_{idx:03d}.jpg")

    candidates = []
    if query_en:
        candidates.append(query_en)
    if query_vn:
        candidates.append(query_vn)

    # Biến thể từ query_en
    variants = []
    for q in candidates:
        if not q:
            continue
        variants.append(q)
        words = re.findall(r'[\wàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+',
                           q, re.IGNORECASE)
        if len(words) >= 4:
            variants.append(" ".join(words[:4]))
        if len(words) >= 2:
            variants.append(" ".join(words[:2]))

    seen, final = set(), []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            final.append(v)

    # Tầng 1: Pexels
    for q in final:
        if fetch_image_pexels(q, p_key, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 2: Wikimedia
    for q in final:
        if fetch_image_wikimedia(q, used_urls, used_hashes, dest):
            return _finalize_image(dest)

    # Tầng 3: DuckDuckGo
    for q in final[:3]:
        if fetch_image_ddg(q, used_urls, used_hashes, dest, is_english):
            return _finalize_image(dest)

    Image.new("RGB", (W, H), (18, 22, 32)).save(dest, "JPEG", quality=88)
    return dest


# ==============================================================================
# FETCH VIDEO B-ROLL
# ==============================================================================
def fetch_broll_clip(query_en, idx, target_frames, p_key, workdir, used_vid_ids, add_motion=False):
    if not p_key:
        return None
    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    dur = target_frames / FPS
    headers = {"Authorization": p_key}

    terms = [query_en]
    words = [w for w in re.findall(r'[a-zA-Z]+', query_en) if len(w) > 2]
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
# AI — PROMPT BẢN CŨ (single query, few-shot đầy đủ)
# ==============================================================================
def ai_extract_query(client, seg_batch, b_start, is_english):
    """
    Trích xuất query_vn + query_en cho mỗi câu thoại — Y HỆT bản cũ.
    Trả về {idx: {"query_vn": ..., "query_en": ...}}
    """
    lines = "\n".join([f"[{i + b_start}] {s['text'][:140]}" for i, s in enumerate(seg_batch)])

    prompt = f"""Bạn là một đạo diễn hình ảnh có khả năng thích ứng linh hoạt tuyệt đối.
Nhiệm vụ: Lắng nghe từng câu thoại và trích xuất ĐÚNG HÀNH ĐỘNG, ĐỊA ĐIỂM, VẬT THỂ được nói đến.

QUY TẮC CỐT LÕI:
1. KHÔNG ÉP BẤT KỲ ĐỊNH KIẾN NÀO:
   - Nói về đi uống cà phê, nhân viên nữ nhìn trộm -> 'coffee shop barista girl smiling customer table'
   - Nói về hai người bạn đi cạnh nhau, so sánh ngoại hình -> 'two young men walking street outdoor candid'
   - Nói về xin chụp ảnh, đưa điện thoại -> 'people taking selfie photo smartphone smiling outdoor'
   - Nói về xe máy, đường mưa -> 'motorcycle road heavy rain'
2. BẮT BUỘC dùng cấu trúc: [ĐỐI TƯỢNG CỤ THỂ] + [HÀNH ĐỘNG THỰC TẾ] + [ĐỊA ĐIỂM / BỐI CẢNH].
3. TUYỆT ĐỐI CẤM dùng các từ cảm xúc mơ hồ ('sad', 'depressed', 'lonely', 'thinking', 'suffering').
4. CẤM chữ viết, quote, logo.

Đoạn thoại:
{lines}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "quán cà phê hai bạn trẻ", "query_en": "modern cafe two young men sitting table coffee"}}
]}}"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.12,
        )
        content = resp.choices[0].message.content
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if not m:
            return {}
        parsed = json.loads(m.group(0)).get("scenes", [])
        result = {}
        for it in parsed:
            if "index" not in it:
                continue
            idx = int(it["index"])
            result[idx] = {
                "query_vn": (it.get("query_vn") or "").strip(),
                "query_en": (it.get("query_en") or "").strip(),
            }
        return result
    except Exception as e:
        log(f"AI extract fail: {str(e)[:120]}")
        return {}


# ==============================================================================
# BGM OPTIONAL
# ==============================================================================
BGM_URLS = {
    "chill": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Scott_Buckley_-_Aurora.mp3",
    "epic":  "https://upload.wikimedia.org/wikipedia/commons/4/4c/Scott_Buckley_-_Aurora.mp3",
    "sad":   "https://upload.wikimedia.org/wikipedia/commons/2/2b/Kai_Engel_-_03_-_Great_Expectations.ogg",
    "happy": "https://upload.wikimedia.org/wikipedia/commons/f/f1/Kai_Engel_-_08_-_Soft.ogg",
}


def fetch_bgm(mood, workdir):
    url = BGM_URLS.get(mood, BGM_URLS["chill"])
    dest = os.path.join(workdir, "bgm.mp3")
    if download_file(url, dest, min_size=50000, timeout=20):
        return dest
    return None


# ==============================================================================
# PIPELINE
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

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            total_audio_dur = probe_duration(audio_path)
            total_required_frames = int(round(total_audio_dur * FPS))
            client = Groq(api_key=GROQ_API_KEY)
            log(f"Audio duration: {total_audio_dur:.2f}s")

            # 1. Whisper
            status.update(label="🎙️ 1/4: Whisper bóc tách timestamp...")
            compressed = os.path.join(workdir, "whisper_input.mp3")
            subprocess.run([
                FFMPEG_EXE, "-y", "-i", audio_path, "-vn",
                "-ar", "16000", "-ac", "1", "-b:a", "48k", compressed
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            data = call_whisper(client, compressed)
            if not data:
                st.error("❌ Không transcribe được. Bật Whisper trong Groq console (tab Audio) "
                         "hoặc thêm `faster-whisper` vào requirements.txt.")
                st.stop()

            raw_segs = data.get("segments") or []
            detected_lang = (data.get("language") or "vietnamese").lower()
            is_english = "en" in detected_lang

            # 2. Chia cảnh dày
            segments = build_segments_from_whisper(raw_segs, total_audio_dur)
            status.write(f"📊 **{len(segments)} cảnh** (~{total_audio_dur / len(segments):.1f}s/cảnh)")

            accumulated = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i + 1]["start"] - segments[i]["start"]
                    segments[i]["target_frames"] = max(15, int(round(seg_dur * FPS)))
                    accumulated += segments[i]["target_frames"]
                else:
                    segments[i]["target_frames"] = max(15, total_required_frames - accumulated)

            # 3. AI bóc tách query (prompt bản cũ)
            status.update(label="🧠 2/4: AI bóc tách hành động & vật thể theo câu thoại...")
            by_idx = {}
            batch_size = 12
            for b_start in range(0, len(segments), batch_size):
                sub = segments[b_start:b_start + batch_size]
                by_idx.update(ai_extract_query(client, sub, b_start, is_english))
                status.write(f"✓ Câu {b_start + 1}–{b_start + len(sub)}")

            # 4. Dựng cảnh
            status.update(label="🎬 3/4: Dựng cảnh với ảnh/video khớp voice...")
            clips_txt = os.path.join(workdir, "clips.txt")
            progress = st.progress(0.0)

            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "hành động đời thực"
                    query_en = sc_data.get("query_en") or "cinematic daily action"
                    t_frames = sc["target_frames"]

                    # Slot video: lẻ (1/2 cảnh) — y bản cũ dùng 1/5, đây dày hơn
                    is_video_slot = (idx % 2 == 1) and (idx != len(segments) - 1)
                    clip_path = None

                    if is_video_slot and PEXELS_API_KEY:
                        clip_path = fetch_broll_clip(
                            query_en, idx, t_frames, PEXELS_API_KEY,
                            workdir, used_vid_ids, add_motion=enable_motion
                        )

                    if not clip_path:
                        img_path = fetch_matching_image(
                            query_vn, query_en, idx, workdir,
                            used_urls, used_img_hashes, PEXELS_API_KEY, is_english
                        )
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")
                    tag = "VIDEO" if is_video_slot and clip_path else "IMG"
                    status.write(f"✓ [{tag}] Cảnh {idx + 1}/{len(segments)}: `{query_en[:55]}`")
                    progress.progress((idx + 1) / len(segments))

            # 5. Xuất master
            status.update(label="⚡ 4/4: Ghép master...")
            out_path = os.path.join(workdir, "output.mp4")

            if enable_bgm:
                bgm_path = fetch_bgm(bgm_choice, workdir)
                if bgm_path:
                    # Mix voice + BGM
                    subprocess.run([
                        FFMPEG_EXE, "-y",
                        "-f", "concat", "-safe", "0", "-i", clips_txt,
                        "-i", audio_path,
                        "-stream_loop", "-1", "-i", bgm_path,
                        "-filter_complex",
                        "[1:a]volume=1.0[a1];"
                        "[2:a]volume=0.12,atrim=0:%d,afade=t=in:d=1.5,afade=t=out:st=%d:d=2[a2];"
                        "[a1][a2]amix=inputs=2:duration=first:dropout_transition=2,"
                        "loudnorm=I=-14:TP=-1.5:LRA=11[aout]" % (
                            int(total_audio_dur),
                            max(0, int(total_audio_dur) - 2)
                        ),
                        "-map", "0:v:0", "-map", "[aout]",
                        "-t", f"{total_audio_dur:.3f}",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        out_path
                    ], check=True)
                else:
                    st.warning("⚠️ Không tải được BGM, xuất video không nhạc.")
                    enable_bgm = False

            if not enable_bgm:
                subprocess.run([
                    FFMPEG_EXE, "-y",
                    "-f", "concat", "-safe", "0", "-i", clips_txt,
                    "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-t", f"{total_audio_dur:.3f}",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    out_path
                ], check=True)

            status.update(label=f"✅ Hoàn thành! {len(segments)} cảnh", state="complete")

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
