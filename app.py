"""
Xưởng Video Diễn Hoạt Kiến Thức AI — Bản Siêu Cấp V10.3
=======================================================
V10.3 FIX:
- Cache transcript theo HASH voice → không lẫn cache giữa các voice khác nhau
- Batch audio có hash tiền tố → không trùng tên batch giữa các lần chạy
- Qwen 14000 tokens (không cắt JSON)
- English mode: tắt hiệu ứng vẽ (hiện ảnh ngay + camera motion)
- English mode: full-frame camera (không title band)
"""

import os, re, io, json, math, time, base64, random, shutil, subprocess, tempfile, threading, wave, hashlib
from pathlib import Path
from queue import Queue, Empty
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
import cv2
import numpy as np

# ============================================================
# CẤU HÌNH
# ============================================================
APP_TITLE = "Xưởng Video Diễn Hoạt Kiến Thức AI (V10.3)"
BATCH_SECONDS = 5 * 60
FPS = 24
WIDTH = 1280
HEIGHT = 720
TITLE_BAND_H = 95
CONTENT_H = HEIGHT - TITLE_BAND_H
REVEAL_RADIUS = 34
DRAW_DURATION_RATIO = 0.45
PHASE_RATIOS = (0.40, 0.35, 0.25)
MAX_TOTAL_FALLBACK_TIME = 300
FAIR_SHARE_MULTIPLIER = 1.5
MAX_ATTEMPTS_PER_SCENE = 3
CIRCUIT_BREAKER_THRESHOLD = 3
FAIL_SLEEP_SECONDS = 2.0
SLOW_PROVIDER_THRESHOLD = 10.0
SLOW_PROVIDER_PENALTY = 0.5
SFX_SAMPLE_RATE = 22050
FONT_DIR = Path.home() / ".fonts"

AGNES_API_URL = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_MODEL = "agnes-image-2.1-flash"
CLOUDFLARE_BASE = "https://api.cloudflare.com/client/v4/accounts/"
CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_MODEL = "black-forest-labs/FLUX.1-schnell"
FREETHEAI_BASE = "https://api.freetheai.xyz/v1/images/generations"
TOGETHER_BASE = "https://api.together.xyz/v1/images/generations"
TOGETHER_MODEL = "black-forest-labs/FLUX.1-schnell-Free"
NEXA_BASE = "https://api.nexa-api.com/v1/images/generations"
NEXA_MODEL = "flux-schnell"
POLLINATIONS_BASE = "https://gen.pollinations.ai/image/"

COLOR_MAP = {"red": "#d32f2f", "green": "#2e7d32", "blue": "#1565c0",
    "orange": "#ef6c00", "purple": "#6a1b9a", "black": "#212121",
    "yellow": "#f9a825", "pink": "#c2185b", "teal": "#00838f", "brown": "#5d4037"}
SIZE_MAP = {"small": 22, "medium": 30, "large": 44, "huge": 58}
VALID_SFX = {"none", "whoosh", "pop", "ding", "impact", "sad", "bell", "typing", "sparkle", "swoosh"}
HORROR_SFX = {"none", "whoosh", "impact", "sad", "typing", "swoosh",
              "creak", "whisper", "scream", "heartbeat", "thunder", "silence_break"}

CHAR_PRESETS = {
    "Custom (tự nhập)": "",
    "🏹 Cổ đại / Cave man": "a cartoon caveman character with simple line art, big expressive eyes, messy wild hair, wearing a rough animal-skin tunic, barefoot, thin stick-figure body with slightly rounded head, hand-drawn doodle style, thick black outlines, funny and expressive face",
    "🐫 Ai Cập cổ": "a cartoon ancient Egyptian character with simple line art, big expressive eyes, wearing a white shendyt kilt and gold collar, black bob hairstyle, thin stick-figure body, thick black outlines, hand-drawn doodle style",
    "⚔️ Trung cổ / Medieval": "a cartoon medieval peasant character with simple line art, big expressive eyes, wearing a rough brown tunic and rope belt, messy hair, thin stick-figure body, thick black outlines, hand-drawn doodle style",
    "🦴 Tiền sử / Prehistoric": "a cartoon prehistoric human character with simple line art, big expressive eyes, long messy hair, wearing a ragged animal fur, barefoot, thin stick-figure body, thick black outlines, hand-drawn doodle style",
    "🔬 Nhà khoa học điên": "a cartoon mad scientist character with simple line art, big expressive eyes, wild messy hair, wearing a white lab coat and goggles on forehead, thin stick-figure body, thick black outlines, hand-drawn doodle style",
    "👑 Vua / Hoàng đế": "a cartoon king character with simple line art, big expressive eyes, wearing a golden crown and purple royal robe with fur trim, thin stick-figure body, thick black outlines, hand-drawn doodle style",
    "🧙 Phù thủy": "a cartoon wizard character with simple line art, big expressive eyes, long white beard, wearing a tall pointed hat and long starry robe, thin stick-figure body, thick black outlines, hand-drawn doodle style",
}

NOTE_FREQ = {
    'C3':130.81,'D3':146.83,'E3':164.81,'F3':174.61,'G3':196.00,'A3':220.00,'B3':246.94,
    'C4':261.63,'D4':293.66,'E4':329.63,'F4':349.23,'G4':392.00,'A4':440.00,'B4':493.88,
    'C5':523.25,'D5':587.33,'E5':659.25,'F5':698.46,'G5':783.99,'A5':880.00,
    'Eb4':311.13,'Ab4':415.30,'Db4':277.18,'Bb3':233.08,'Eb3':155.56,'Bb4':466.16,
    'B2':123.47,'F3':174.61,'Ab3':207.65,'D3':146.83,'G2':98.00,
}
EMOTION_CHORDS = {
    "happy":         [('C4','E4','G4'),('F4','A4','C5'),('G4','B4','D5'),('C5','E5','G5')],
    "sad":           [('A3','C4','E4'),('F3','A3','C4'),('D4','F4','A4'),('E4','G4','B4')],
    "epic":          [('C3','G3','C4'),('G3','D4','G4'),('A3','E4','A4'),('F3','C4','F4')],
    "calm":          [('C4','E4','G4'),('A3','C4','E4'),('F3','A3','C4'),('G3','B3','D4')],
    "tense":         [('C4','Eb4','G4'),('Db4','F4','Ab4'),('C4','Eb4','G4'),('Bb3','D4','F4')],
    "inspirational": [('C4','E4','G4'),('A3','C4','E4'),('F3','A3','C4'),('G3','B3','D4')],
    "neutral":       [('C4','E4','G4'),('C4','E4','G4'),('F4','A4','C5'),('G4','B4','D5')],
    "dread":         [('A3','C4','E4'),('G3','Bb3','D4'),('F3','Ab3','C4'),('E3','G3','B3')],
    "panic":         [('D4','F4','A4'),('Eb4','G4','Bb4'),('D4','F4','A4'),('C4','Eb4','G4')],
    "eerie":         [('C3','Eb3','G3'),('Db3','F3','Ab3'),('C3','Eb3','G3'),('B2','D3','F3')],
    "ominous":       [('C3','G3','C4'),('Bb3','F4','Bb4'),('Ab3','Eb4','Ab4'),('G3','D4','G4')],
    "mysterious":    [('A3','C4','E4'),('F3','A3','C4'),('D4','F4','A4'),('E4','G4','B4')],
}
VALID_EMOTIONS = set(EMOTION_CHORDS.keys()) | {"none"}

# ============================================================
# HORROR SANITIZE
# ============================================================
HORROR_SANITIZE = {
    r"\bblood splatter\b": "crimson liquid splatter",
    r"\bbloodbath\b": "crimson scene",
    r"\bbloody\b": "crimson-stained",
    r"\bblood\b": "crimson liquid",
    r"\bbleed(ing)?\b": "dripping red",
    r"\bdead body\b": "still silhouette",
    r"\bcorpse\b": "motionless figure",
    r"\bdead\b": "motionless",
    r"\bdeath\b": "final moment",
    r"\bdied\b": "became still",
    r"\bkilled\b": "silenced",
    r"\bkill(ing|er)?\b": "confrontation",
    r"\bmurder(er)?\b": "dark event",
    r"\bsuicide\b": "despair",
    r"\bweapon\b": "sharp object",
    r"\bknife\b": "gleaming blade",
    r"\bblade\b": "sharp edge",
    r"\bgun\b": "dark shape",
    r"\bgore\b": "dramatic red accents",
    r"\bwound(ed)?\b": "red mark",
    r"\bstab(bing|bed)?\b": "sharp impact",
    r"\bmutilat(e|ed|ion)\b": "damaged",
    r"\bdismember(ed)?\b": "broken apart",
    r"\bsever(ed)?\b": "cut apart",
    r"\bchop(ped)?\b": "cut",
    r"\bviolent\b": "intense",
    r"\bviolence\b": "intense confrontation",
    r"\btorture(d)?\b": "torment",
    r"\bscream(ing)?\b": "cry of fear",
    r"\bterror\b": "deep dread",
}

def horror_sanitize(text):
    result = text
    for pattern, rep in HORROR_SANITIZE.items():
        result = re.sub(pattern, rep, result, flags=re.IGNORECASE)
    return result

# ============================================================
# CACHE HELPERS — V10.3
# ============================================================
def file_hash(path, nbytes=10000):
    """MD5 hash của 10KB đầu file — dùng để phân biệt các voice khác nhau."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(nbytes)).hexdigest()[:12]
    except Exception:
        return hashlib.md5(str(path).encode()).hexdigest()[:12]

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title("🎬 Xưởng Video Diễn Hoạt Kiến Thức AI — V10.3")
st.caption("Cache theo hash voice + Qwen 14k + English full-frame + sticker title")

with st.sidebar:
    st.header("🎨 Style Mode")
    style_mode_ui = st.radio("Phong cách video",
        ["📚 Kiến Thức (Comic)", "👻 Kinh Dị (Horror)"], index=0)
    style_mode = "horror" if "Horror" in style_mode_ui else "comic"

    st.header("🌐 Ngôn ngữ")
    language_mode = st.selectbox("Ngôn ngữ video",
        ["Auto Detect", "Tiếng Việt", "English"], index=0,
        help="English: AI vẽ sticker title + full-frame camera. Vietnamese: Pillow overlay.")

    effective_lang = "vi"
    if language_mode == "English": effective_lang = "en"
    elif language_mode == "Tiếng Việt": effective_lang = "vi"

    if style_mode == "comic":
        if effective_lang == "en":
            st.success("🇬🇧 English: AI vẽ sticker title + full-frame camera")
        else:
            st.info("🇻🇳 Vietnamese: Pillow overlay + title band 95px")
    else:
        st.warning("⚠️ Horror mode: nhịp 5-10s/cảnh. Khuyên dùng Pollinations flux-pro.")

    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""), type="password")

    with st.expander("🎨 Nhà cung cấp ảnh AI", expanded=False):
        pollinations_key = st.text_input("Pollinations API Key",
            value=os.getenv("POLLINATIONS_API_KEY", ""), type="password")
        pollinations_model = st.selectbox("Pollinations Model",
            ["flux-pro", "flux", "gptimage", "kontext", "flux-realism"], index=0)
        agnes_key = st.text_input("Agnes AI API Key",
            value=os.getenv("AGNES_API_KEY", ""), type="password")
        cf_account = st.text_input("Cloudflare Account ID",
            value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""), type="password")
        cf_token = st.text_input("Cloudflare API Token",
            value=os.getenv("CLOUDFLARE_API_TOKEN", ""), type="password")
        hf_token = st.text_input("Hugging Face Token",
            value=os.getenv("HF_TOKEN", ""), type="password")
        freetheai_key = st.text_input("FreeTheAi API Key",
            value=os.getenv("FREETHEAI_API_KEY", ""), type="password")
        together_key = st.text_input("Together AI API Key",
            value=os.getenv("TOGETHER_API_KEY", ""), type="password")
        nexa_key = st.text_input("NexaAPI Key",
            value=os.getenv("NEXA_API_KEY", ""), type="password")

    st.header("📝 Văn bản kịch bản (tùy chọn)")
    script_text = st.text_area("Dán kịch bản", value="", height=100,
        help="⚠️ Combined mode có thể lệch timing 5-20%. Dùng 'Chỉ dùng voice' để chính xác nhất.")
    use_script_mode = st.radio("Chế độ phân tích",
        ["Chỉ dùng voice", "Kết hợp voice + text", "Chỉ dùng text"], index=0,
        help="'Chỉ dùng voice' = timing chính xác 100% (khuyên dùng).")

    st.header("👥 Nhân vật")
    char_main_name = st.text_input("Tên nhân vật chính", value="Tôi")
    char_main_desc = st.text_area("Mô tả ngoại hình (English)",
        value="a young Vietnamese man, short black hair, brown eyes, wearing a blue hoodie and dark jeans",
        height=60)
    char_second_name = st.text_input("Tên nhân vật phụ", value="Linh")
    char_second_desc = st.text_area("Mô tả ngoại hình (English)",
        value="a young Vietnamese woman, long black hair tied in ponytail, wearing an orange hoodie",
        height=60)
    enable_char_lock = st.checkbox("🔒 Khóa ngoại hình theo tên", value=True)
    enable_seed_lock = st.checkbox("🎲 Cố định Seed", value=False)

    st.markdown("---")
    st.subheader("🌍 Global Character Lock")
    enable_global_char = st.checkbox("Bật Global Lock", value=False)
    preset_choice = st.selectbox("🎨 Preset nhân vật", list(CHAR_PRESETS.keys()), index=1)
    if preset_choice != "Custom (tự nhập)" and CHAR_PRESETS[preset_choice]:
        default_global = CHAR_PRESETS[preset_choice]
    else:
        default_global = "a young Vietnamese man, short black hair, brown eyes, wearing a blue hoodie"
    global_char_desc = st.text_area("Mô tả nhân vật chính toàn cục (English)",
        value=default_global, height=100)

    st.header("🔊 Âm thanh")
    enable_sfx = st.checkbox("Bật sound effects", value=True)
    sfx_volume = st.slider("Âm lượng SFX (dB)", -30, 0, -12)
    enable_music = st.checkbox("🎵 Nhạc nền theo cảm xúc", value=True)
    music_volume = st.slider("Âm lượng nhạc nền (dB)", -35, -10, -22)

    st.header("🧠 Groq Model")
    stt_model = st.selectbox("STT Model", ["whisper-large-v3", "whisper-large-v3-turbo"], index=0)
    planner_model = st.selectbox("Biên kịch Model",
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=1)

    st.header("🎬 Phong cách diễn hoạt")
    draw_style = st.selectbox("Render style", [
        "1. Vẽ 3 phase + Pan/Zoom",
        "2. Hybrid (Tay vẽ + Steadicam)",
        "3. Chỉ Camera Pan & Zoom",
        "4. Bảng trắng cổ điển",
    ], index=0)

    st.header("🎨 Overlay")
    enable_rich_overlay = st.checkbox("Overlay nhiều text box", value=True,
        help="Chỉ áp dụng cho Vietnamese mode")
    enable_arrows = st.checkbox("Vẽ mũi tên", value=True)
    enable_shadow = st.checkbox("Đổ bóng chữ", value=True)

    st.header("🎥 Camera")
    camera_motion_mode = st.selectbox("Camera motion", [
        "Auto (AI chọn)", "Random",
        "Cố định: zoom_in_center", "Cố định: zoom_out_center",
        "Cố định: pan_left_to_right", "Cố định: pan_right_to_left",
        "Cố định: ken_burns_slow", "Cố định: static",
    ], index=0)

    st.header("⏱️ Nhịp cảnh")
    if style_mode == "horror":
        default_min, default_max = 6, 10
    else:
        default_min, default_max = 19, 27
    scene_min = st.slider("Tối thiểu (giây)", 5, 25, default_min)
    scene_max = st.slider("Tối đa (giây)", 10, 35, default_max)
    if scene_max < scene_min: scene_max = scene_min

    st.header("⚙️ Khác")
    max_scenes = st.slider("Số cảnh tối đa/batch", 5, 60,
        35 if style_mode == "horror" else 25)
    image_timeout = st.slider("Timeout ảnh (giây)", 20, 90, 45)
    flux_steps = st.slider("Số bước FLUX", 4, 8, 4)
    fair_share_enabled = st.checkbox("Fair share cap", value=True)
    circuit_breaker_enabled = st.checkbox("Circuit breaker", value=True)
    prioritize_fast = st.checkbox("⚡ Ưu tiên provider nhanh", value=True)

# ============================================================
# UTILITIES
# ============================================================
def run_cmd(cmd, timeout=600):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0: raise RuntimeError(p.stderr[-5000:] or "Lệnh thất bại")
    return p.stdout

def ffprobe_duration(path):
    out = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60)
    return float(out.strip())

def extract_json(text):
    if not text: raise ValueError("Empty response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try: return json.loads(text)
    except Exception: pass
    best = None; blen = 0; depth = 0; start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    cand = text[start:i+1]
                    if len(cand) > blen:
                        try:
                            obj = json.loads(cand); best = obj; blen = len(cand)
                        except Exception: pass
    if best is not None: return best
    raise ValueError(f"JSON invalid (preview={text[:200]})")

def groq_client(key): return Groq(api_key=key)

def transcribe_file(client, path, model, language=None, cache_dir=None):
    # V10.3: Cache theo HASH của file audio → không lẫn giữa các voice khác nhau
    if cache_dir:
        fh = file_hash(path)
        cf = Path(cache_dir) / f"{fh}_transcript.json"
        if cf.exists():
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
                st.info(f"💾 Cache hit ({fh}): {Path(path).name}")
                class CR:
                    def __init__(s, d): s._d = d
                    def model_dump(s): return s._d
                    def __getattr__(s, k): return s._d.get(k)
                return CR(data)
            except Exception: pass
    with open(path, "rb") as f:
        kw = {"file": (Path(path).name, f.read()), "model": model,
              "response_format": "verbose_json", "timestamp_granularities": ["segment"],
              "temperature": 0.0}
        if language: kw["language"] = language
        result = client.audio.transcriptions.create(**kw)
    if cache_dir:
        try:
            data = result.model_dump() if hasattr(result, "model_dump") else result
            fh = file_hash(path)
            (Path(cache_dir) / f"{fh}_transcript.json").write_text(
                json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception: pass
    return result

def detect_language(result):
    try:
        d = result.model_dump() if hasattr(result, "model_dump") else result
        return d.get("language", "unknown") if isinstance(d, dict) else getattr(result, "language", "unknown")
    except Exception: return "unknown"

def chunk_audio(src, out_dir, bs):
    # V10.3: Hash tiền tố tránh trùng batch giữa các voice
    vh = file_hash(src)
    pattern = str(Path(out_dir) / f"{vh}_batch_%03d.m4a")
    run_cmd(["ffmpeg", "-y", "-i", str(src), "-map", "0:a:0", "-c:a", "aac", "-b:a", "96k",
             "-f", "segment", "-segment_time", str(bs), "-reset_timestamps", "1", pattern], timeout=900)
    return sorted(Path(out_dir).glob(f"{vh}_batch_*.m4a"))

def normalize_segments(result, offset):
    data = result.model_dump() if hasattr(result, "model_dump") else result
    segments = data.get("segments", []) if isinstance(data, dict) else getattr(result, "segments", [])
    out = []
    for s in segments or []:
        if isinstance(s, dict):
            st_ = float(s.get("start", 0)); e_ = float(s.get("end", st_)); tx = str(s.get("text", "")).strip()
        else:
            st_ = float(getattr(s, "start", 0)); e_ = float(getattr(s, "end", st_)); tx = str(getattr(s, "text", "")).strip()
        if tx: out.append({"start": st_ + offset, "end": e_ + offset, "text": tx})
    return out

def sanitize_prompt_text(prompt):
    for pat, rep in {
        r"\bblood\b": "dark ink", r"\bbleed\b": "drip", r"\bsuicide\b": "despair",
        r"\bkill(ing|er)?\b": "oppression", r"\bdead\b": "fallen", r"\bdeath\b": "crisis",
        r"\bcorpse\b": "shadow", r"\bjump(ing)?\b": "falling shadow", r"\bweapon\b": "heavy chain",
    }.items():
        prompt = re.sub(pat, rep, prompt, flags=re.IGNORECASE)
    return prompt

def build_character_lock(mn, md, sn, sd, enabled=True):
    if not enabled: return {}
    lock = {}
    if mn.strip() and md.strip(): lock[mn.strip().lower()] = md.strip()
    if sn.strip() and sd.strip(): lock[sn.strip().lower()] = sd.strip()
    return lock

def enforce_character_lock(prompt, char_lock):
    if not char_lock: return prompt
    pl = prompt.lower()
    result = prompt
    global_desc = char_lock.get("__global__")
    if global_desc:
        result = f"[MAIN CHARACTER (MUST appear in this scene): {global_desc}] " + result
    for name, desc in char_lock.items():
        if name == "__global__": continue
        if name in pl and desc not in pl:
            result = f"[CHARACTER: {name} = {desc}] " + result
    return result

# ============================================================
# TITLE BAND
# ============================================================
def split_title_band(img_bgr): return img_bgr[:TITLE_BAND_H, :].copy(), img_bgr[TITLE_BAND_H:, :].copy()

def compose_frame(tb, ct):
    c = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    c[:TITLE_BAND_H] = tb; c[TITLE_BAND_H:] = ct
    return c

def crop_content_motion(c, scale, cx, cy):
    ch, cw = c.shape[:2]
    w = max(1, min(cw, int(cw / max(0.5, scale))))
    h = max(1, min(ch, int(ch / max(0.5, scale))))
    x1 = max(0, min(cw - w, int(cx - w / 2)))
    y1 = max(0, min(ch - h, int(cy - h / 2)))
    return cv2.resize(c[y1:y1+h, x1:x1+w], (cw, ch), interpolation=cv2.INTER_LINEAR)

def crop_full_frame(frame_bgr, scale, cx, cy):
    ch, cw = frame_bgr.shape[:2]
    w = max(1, min(cw, int(cw / max(0.5, scale))))
    h = max(1, min(ch, int(ch / max(0.5, scale))))
    x1 = max(0, min(cw - w, int(cx - w / 2)))
    y1 = max(0, min(ch - h, int(cy - h / 2)))
    return cv2.resize(frame_bgr[y1:y1+h, x1:x1+w], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

def traj_to_content(traj_full):
    out = [(int(px), int(py - TITLE_BAND_H)) for (px, py) in traj_full if py >= TITLE_BAND_H]
    return out if out else [(WIDTH // 2, CONTENT_H // 2)]

def split_traj_phases(traj, ratios=PHASE_RATIOS):
    n = len(traj)
    if n < 3: return [traj, [], []]
    b1 = max(1, int(n * ratios[0]))
    b2 = max(b1 + 1, int(n * (ratios[0] + ratios[1])))
    b2 = min(b2, n - 1)
    return [traj[:b1], traj[b1:b2], traj[b2:]]

# ============================================================
# FONT
# ============================================================
FONT_URLS = {
    "NotoSans-Bold.ttf": [
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
        "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf/NotoSans/NotoSans-Bold.ttf",
    ],
}

@st.cache_resource(show_spinner=False)
def _ensure_fonts():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, urls in FONT_URLS.items():
        t = FONT_DIR / fname
        if t.exists() and t.stat().st_size > 50000: continue
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 50000:
                    t.write_bytes(r.content); break
            except Exception: continue
    tmp = Path("/tmp/DejaVuSans-Bold.ttf")
    if not tmp.exists():
        try:
            r = requests.get("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf", timeout=30)
            if r.status_code == 200 and len(r.content) > 100000: tmp.write_bytes(r.content)
        except Exception: pass
    return True

def font_for(size, bold=True):
    try: _ensure_fonts()
    except Exception: pass
    noto = FONT_DIR / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf")
    if noto.exists():
        try: return ImageFont.truetype(str(noto), size)
        except Exception: pass
    for p in [
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: continue
    for fb in ["arial.ttf", "DejaVuSans.ttf"]:
        try: return ImageFont.truetype(fb, size)
        except Exception: continue
    tmp = Path("/tmp/DejaVuSans-Bold.ttf")
    if tmp.exists():
        try: return ImageFont.truetype(str(tmp), size)
        except Exception: pass
    return ImageFont.load_default()

# ============================================================
# SFX + MUSIC
# ============================================================
def generate_sfx_wav(sfx_type, duration=0.5, sr=SFX_SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), False)
    if sfx_type == "whoosh":
        d = np.random.randn(len(t)) * np.exp(-t * 4) * np.sin(np.pi * t / duration) * 0.4
    elif sfx_type == "pop":
        d = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 20) * 0.6
    elif sfx_type == "ding":
        d = np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 3) * 0.5
    elif sfx_type == "impact":
        d = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 8) * 0.8
    elif sfx_type == "sad":
        d = np.sin(2 * np.pi * (400 - 250 * (t / duration)) * t) * np.exp(-t * 2) * 0.5
    elif sfx_type == "bell":
        d = (np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)) * np.exp(-t * 4) * 0.5
    elif sfx_type == "typing":
        d = np.zeros_like(t)
        for i in range(int(duration * 12)):
            idx = int(i * sr * (duration / 12))
            if idx < len(d) - 80: d[idx:idx+80] += np.random.randn(80) * 0.4
    elif sfx_type == "sparkle":
        fr = [1500, 2000, 2500, 3000]
        d = sum(np.sin(2 * np.pi * f * t) for f in fr) / len(fr) * np.exp(-t * 2.5) * 0.4
    elif sfx_type == "swoosh":
        d = np.random.randn(len(t)) * (t / duration) * np.exp(-t * 3) * 0.4
    elif sfx_type == "creak":
        fr = 200 + 50 * np.sin(2 * np.pi * 2 * t)
        d = np.sin(2 * np.pi * fr * t) * (0.3 + 0.7 * np.random.rand(len(t))) * np.exp(-t * 2) * 0.5
    elif sfx_type == "whisper":
        noise = np.random.randn(len(t))
        env = np.sin(np.pi * t / duration) * 0.6
        d = noise * env * 0.3
    elif sfx_type == "scream":
        fr = 800 + 600 * np.sin(2 * np.pi * 6 * t)
        d = np.sin(2 * np.pi * fr * t) * np.exp(-t * 1.5) * 0.6
    elif sfx_type == "heartbeat":
        d = np.zeros_like(t)
        for i in range(int(duration * 1.5)):
            idx = int(i * sr / 1.5)
            if idx < len(d) - 200:
                thump = np.sin(2 * np.pi * 60 * np.arange(200) / sr) * np.exp(-np.arange(200) / 50)
                d[idx:idx+200] += thump * 0.7
    elif sfx_type == "thunder":
        noise = np.random.randn(len(t))
        env = np.exp(-t * 1.2) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        d = noise * env * 0.6
    elif sfx_type == "silence_break":
        d = np.zeros_like(t)
        idx = int(len(t) * 0.7)
        d[idx:] = np.sin(2 * np.pi * 1500 * np.arange(len(t)-idx) / sr) * np.exp(-np.arange(len(t)-idx) / (sr * 0.05)) * 0.5
    else: return None
    mx = np.max(np.abs(d))
    if mx > 0: d = d / mx * 0.5
    return d.astype(np.float32)

def write_wav(data, path, sr=SFX_SAMPLE_RATE):
    di = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(di.tobytes())

def generate_music_track(emotion, duration, sr=SFX_SAMPLE_RATE):
    chords = EMOTION_CHORDS.get(emotion, EMOTION_CHORDS["neutral"])
    total = int(duration * sr)
    if total <= 0: return np.zeros(0, dtype=np.float32)
    track = np.zeros(total, dtype=np.float32)
    cd = 2.5; spc = int(cd * sr)
    nch = max(1, int(np.ceil(duration / cd)))
    t_env = np.linspace(0, 1, spc)
    att = np.minimum(t_env * 8, 1.0); dec = np.exp(-t_env * 1.2); rel = np.minimum((1 - t_env) * 8, 1.0)
    env = (att * dec * rel * 0.6).astype(np.float32)
    for i in range(nch):
        chord = chords[i % len(chords)]
        s = i * spc; e = min(s + spc, total); ln = e - s
        if ln <= 0: break
        t = np.linspace(0, ln / sr, ln, False)
        w = np.zeros(ln, dtype=np.float32)
        for note in chord:
            f = NOTE_FREQ.get(note, 261.63)
            vib = 1 + 0.003 * np.sin(2 * np.pi * 5 * t)
            w += np.sin(2 * np.pi * f * vib * t).astype(np.float32)
        w /= len(chord); w *= env[:ln]
        track[s:e] += w
    ds = int(0.15 * sr); rv = np.zeros_like(track)
    for i in range(ds, len(track)): rv[i] = track[i] + 0.3 * rv[i - ds]
    track = track * 0.7 + rv * 0.3
    mx = np.max(np.abs(track))
    if mx > 0: track = track / mx * 0.4
    return track.astype(np.float32)

def build_sfx_track(scenes, out, sr=SFX_SAMPLE_RATE, vol_db=-12):
    td = max((s["end"] for s in scenes), default=0) + 1.0
    ts = int(td * sr); track = np.zeros(ts, dtype=np.float32)
    cache = {}; any_ = False
    for s in scenes:
        t = s.get("sfx", "none")
        if t in ("none", None): continue
        if t not in cache: cache[t] = generate_sfx_wav(t, 0.6, sr)
        d = cache[t]
        if d is None: continue
        st_ = int(s["start"] * sr); e_ = min(st_ + len(d), ts)
        if st_ >= ts: continue
        track[st_:e_] += d[:e_ - st_]; any_ = True
    if not any_: return None
    mx = np.max(np.abs(track))
    if mx > 1.0: track = track / mx
    track *= 10 ** (vol_db / 20.0)
    write_wav(track, out, sr); return out

def build_music_track(scenes, out, sr=SFX_SAMPLE_RATE, vol_db=-22):
    if not scenes: return None
    td = max(s["end"] for s in scenes) + 0.5
    ts = int(td * sr); full = np.zeros(ts, dtype=np.float32)
    cache = {}; any_ = False
    for s in scenes:
        emo = s.get("music_emotion", "neutral")
        if emo in ("none", None): continue
        st_ = s["start"]; dur = s["end"] - s["start"]
        key = f"{emo}_{int(dur)}"
        if key not in cache: cache[key] = generate_music_track(emo, dur, sr)
        mus = cache[key]
        if len(mus) == 0: continue
        ss = int(st_ * sr); se = min(ss + len(mus), ts)
        if ss >= ts: continue
        ln = se - ss; fs = int(0.5 * sr)
        fi = np.minimum(np.arange(ln) / fs, 1.0); fo = np.minimum((ln - np.arange(ln)) / fs, 1.0)
        env = np.minimum(fi, fo).astype(np.float32)
        full[ss:se] += mus[:ln] * env; any_ = True
    if not any_ or np.max(np.abs(full)) == 0: return None
    full *= 10 ** (vol_db / 20.0)
    write_wav(full, out, sr); return out

def mix_audio_tracks(voice, sfx, music, out):
    inputs = ["-i", str(voice)]; filters = ["[0:a]"]; n = 1
    if sfx and Path(sfx).exists(): inputs += ["-i", str(sfx)]; filters.append(f"[{n}:a]"); n += 1
    if music and Path(music).exists(): inputs += ["-i", str(music)]; filters.append(f"[{n}:a]"); n += 1
    if n == 1: shutil.copy2(voice, out); return out
    mix = "".join(filters) + f"amix=inputs={n}:duration=first:dropout_transition=2[aout]"
    run_cmd(["ffmpeg", "-y"] + inputs + ["-filter_complex", mix, "-map", "[aout]",
             "-c:a", "aac", "-b:a", "128k", str(out)], timeout=600)
    return out

# ============================================================
# SCENE PLANNER
# ============================================================
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model,
                    min_s, max_s, max_scenes, camera_mode="auto",
                    language="vi", enable_rich=True, char_lock=None,
                    enable_sfx=True, enable_music=True,
                    user_script="", use_script_mode="voice_only", style_mode="comic"):
    avg_dur = (min_s + max_s) / 2.0
    expected = max(1, round(batch_duration / avg_dur))
    is_en = (language == "en")
    lang_name = "English" if is_en else "Tiếng Việt"
    is_horror = (style_mode == "horror")
    native_text = is_en and not is_horror

    char_note = ""
    if char_lock:
        items = [(n, d) for n, d in char_lock.items() if n != "__global__"]
        if items:
            char_note = "NHÂN VẬT CỐ ĐỊNH:\n" + "\n".join(f"  - {n}: {d}" for n, d in items) + "\n"
        if "__global__" in char_lock:
            char_note += f"\n⭐ NHÂN VẬT CHÍNH (PHẢI xuất hiện trong MỌI cảnh, mô tả y hệt): {char_lock['__global__']}\n"

    if is_horror:
        style_rule = """RÀNG BUỘC HORROR: Dark atmospheric, cinematic, deep shadows. NO text."""
        title_rule = f"- {'English: 3-6 words, mysterious, UPPERCASE' if is_en else 'Tiếng Việt: 3-6 từ, bí ẩn, VIẾT HOA'}"
        visual_rule = """MỖI CẢNH LÀ SÂN KHẤU KINH DỊ KHÁC NHAU."""
        rich_note = """OVERLAY: 1-2 text_boxes. 4 góc.""" if enable_rich else ""
        sfx_note = """SFX: creak/whisper/scream/heartbeat/thunder/silence_break/whoosh/impact/swoosh/none.""" if enable_sfx else ""
        music_note = """NHẠC: dread/panic/eerie/ominous/sad/tense/neutral/none.""" if enable_music else ""
        camera_rule = """slow_zoom_in/slow_zoom_out/creepy_pan_left/creepy_pan_right/dramatic_zoom_face/static_dread."""
    else:
        if native_text:
            style_rule = """RÀNG BUỘC ENGLISH COMIC:
- COLORED cartoon with WARM EARTH TONES (brown, tan, orange, yellow, beige)
- Soft textures, warm firelight, cinematic lighting
- NOT white background
- Thick black outlines, hand-drawn doodle style
- Title as STICKER-STYLE: bold letters + thick outline, NO rectangular banner
- Wide 16:9 cinematic composition"""
        else:
            style_rule = "RÀNG BUỘC: 2D comic doodle, nét mực đen dày, nền TRẮNG TINH, KHÔNG chữ/số."
        title_rule = f"- {'English: 3-6 words, UPPERCASE' if is_en else 'Tiếng Việt: 3-6 từ, VIẾT HOA'}"
        visual_rule = """MỖI CẢNH LÀ SÂN KHẤU KHÁC NHAU. KHÔNG lặp bố cục.
Bao gồm: nhân vật + tư thế, hành động, bối cảnh, đồ vật ẩn dụ, cảm xúc, màu nhấn."""
        rich_note = """OVERLAY: 2-3 text_boxes. 4 góc. Text NGẮN.""" if enable_rich else ""
        sfx_note = """SFX: whoosh/pop/ding/impact/sad/bell/typing/sparkle/swoosh/none.""" if enable_sfx else ""
        music_note = """NHẠC: happy/sad/epic/calm/tense/inspirational/neutral/none.""" if enable_music else ""
        camera_rule = """zoom_in_center/zoom_out_center/pan_left_to_right/pan_right_to_left/zoom_in_top_left/zoom_in_bottom_right/ken_burns_slow/static."""

    sample_motion = "slow_zoom_in" if is_horror else "zoom_in_center"
    sample_sfx = "creak" if is_horror else "sparkle"
    sample_music = "dread" if is_horror else "inspirational"

    if native_text:
        native_note = """
⭐ NATIVE TEXT: Mô tả ngắn gọn vị trí title + callout trong visual_prompt.
Ví dụ: "... Title 'XXX' as sticker text top center. Speech bubble 'YYY' near character."
"""
    else:
        native_note = """QUY TẮC NHÂN VẬT: Ghi rõ "male character"/"female character"."""

    system = f"""Bạn là giám đốc sáng tạo kịch bản cho kênh {("KINH DỊ" if is_horror else "hoạt họa kiến thức")}.
NGÔN NGỮ OUTPUT: {lang_name}.
Nhiệm vụ: Chia đoạn âm thanh {batch_duration:.0f}s thành khoảng {expected} cảnh ({min_s}-{max_s}s/cảnh).

{char_note}
QUY TẮC TIÊU ĐỀ ("title"):
{title_rule}

QUY TẮC CHỮ TRÊN TRANH ("callout_type", "callout_text"):
- "speech": bong bóng thoại. "thought": đám mây. "sticker": nhãn dán. "none": không chữ.

QUY TẮC "visual_prompt":
{visual_rule}

{native_note}

{style_rule}
{rich_note}
{sfx_note}
{music_note}
QUY TẮC CAMERA:
{camera_rule}

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0, "end": {min_s}.0,
      "title": "{'THE CALL' if is_en and is_horror else 'CUỘC GỌI' if is_horror else 'FEAR' if is_en else 'NỖI SỢ'}",
      "callout_type": "speech", "callout_text": "{'WHO?' if is_en else 'AI ĐÓ?'}", "callout_side": "right",
      "camera_motion": "{sample_motion}",
      "sfx": "{sample_sfx}",
      "music_emotion": "{sample_music}",
      "visual_prompt": "2D illustration: ...",
      "text_boxes": [{{"text":"...","x":0.10,"y":0.28,"color":"red","size":"medium","style":"outlined","arrow":null}}]
    }}
  ]
}}
"""

    if use_script_mode == "combined" and user_script.strip():
        user = (f"Audio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\n"
                f"USER SCRIPT (accurate content):\n{user_script}\n\n"
                f"WHISPER TIMING (use ONLY for start/end timestamps):\n{transcript_text}\n\n"
                f"CRITICAL: Scene start/end MUST match Whisper segment timestamps. Do NOT invent timestamps.")
    elif use_script_mode == "text_only" and user_script.strip():
        user = f"Audio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\nSCRIPT:\n{user_script}"
    else:
        user = f"Audio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\nTRANSCRIPT:\n{transcript_text}"

    # V10.3: Qwen cap 14000
    MODEL_CAP = {"qwen/qwen3.8-27b": 30000, "openai/gpt-oss-120b": 40000, "openai/gpt-oss-20b": 9000}
    dyn_max = min(MODEL_CAP.get(model, 40000), max(50000, int(expected * 600 * 1.3)))

    raw = ""
    try:
        r = client.chat.completions.create(model=model, temperature=0.15, max_tokens=dyn_max,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = r.choices[0].message.content or ""
        obj = extract_json(raw); raw_scenes = obj.get("scenes", [])
    except Exception as e:
        st.error(f"❌ Qwen fail: {str(e)[:200]}")
        if raw: st.code(raw[:1000], language="text")
        raw_scenes = []

    valid_motions = (set(HORROR_MOTIONS.keys()) if is_horror else
                     {"zoom_in_center", "zoom_out_center", "pan_left_to_right", "pan_right_to_left",
                      "zoom_in_top_left", "zoom_in_bottom_right", "ken_burns_slow", "static"})
    valid_colors = set(COLOR_MAP.keys())
    valid_sizes = {"small", "medium", "large", "huge"}
    valid_styles = {"plain", "outlined", "highlighted"}
    valid_sfx_set = HORROR_SFX if is_horror else VALID_SFX

    def clean_tbs(tbs):
        out = []
        for tb in (tbs or [])[:4]:
            try:
                text = str(tb.get("text", "")).strip()
                if not text or len(text) > 40: continue
                x = max(0.10, min(0.85, float(tb.get("x", 0.5))))
                y = max(0.28, min(0.88, float(tb.get("y", 0.5))))
                c = str(tb.get("color", "black")).lower()
                if c not in valid_colors: c = "black"
                s = str(tb.get("size", "medium")).lower()
                if s not in valid_sizes: s = "medium"
                stl = str(tb.get("style", "outlined")).lower()
                if stl not in valid_styles: stl = "outlined"
                arr = tb.get("arrow"); ac = None
                if isinstance(arr, dict):
                    try:
                        ac = {"to_x": max(0.05, min(0.95, float(arr.get("to_x", x)))),
                              "to_y": max(0.15, min(0.95, float(arr.get("to_y", y))))}
                    except Exception: ac = None
                out.append({"text": text, "x": x, "y": y, "color": c, "size": s, "style": stl, "arrow": ac})
            except Exception: continue
        return out

    clean = []
    for s in raw_scenes[:max_scenes]:
        try:
            a = max(0.0, float(s["start"])); b = min(batch_duration, float(s["end"]))
            if b <= a + 1.0: continue
            vp = str(s.get("visual_prompt", "")).strip()
            if not vp or len(vp) < 10: continue
            if is_horror: vp = horror_sanitize(vp)
            elif not native_text: vp = sanitize_prompt_text(vp)
            ct = str(s.get("callout_type", "speech")).strip().lower()
            if ct not in ("speech", "thought", "sticker", "none"): ct = "speech"
            cm = str(s.get("camera_motion", list(valid_motions)[0])).strip().lower()
            if cm not in valid_motions: cm = list(valid_motions)[0]
            sfx = str(s.get("sfx", "none")).strip().lower()
            if sfx not in valid_sfx_set: sfx = "none"
            emo = str(s.get("music_emotion", "neutral")).strip().lower()
            if emo not in VALID_EMOTIONS: emo = "neutral"
            clean.append({"start": a, "end": b,
                "title": str(s.get("title", "BÀI HỌC")).strip().upper(),
                "callout_type": ct, "callout_text": str(s.get("callout_text", "")).strip(),
                "callout_side": str(s.get("callout_side", "right")).strip().lower(),
                "camera_motion": cm, "sfx": sfx, "music_emotion": emo,
                "visual_prompt": vp, "text_boxes": clean_tbs(s.get("text_boxes", []))})
        except Exception: continue

    if not clean:
        n = max(3, int(batch_duration / avg_dur)); sd = batch_duration / n
        fb_t = "SCENE" if is_en else "CẢNH"
        fb_prompts = ["Colored cartoon illustration with warm earth tones, a character in a natural scene",
                      "Colored cartoon illustration, dramatic lighting, a character"]
        clean = []
        for i in range(n):
            clean.append({"start": i * sd, "end": (i + 1) * sd, "title": f"{fb_t} {i+1:02d}",
                "callout_type": "none", "callout_text": "", "callout_side": "right",
                "camera_motion": random.choice(list(valid_motions)),
                "sfx": random.choice(list(valid_sfx_set - {"none"})),
                "music_emotion": random.choice(list(VALID_EMOTIONS - {"none"})),
                "visual_prompt": fb_prompts[i % len(fb_prompts)], "text_boxes": []})

    merge_threshold = max(min_s * 0.7, 5.0)
    merged = []
    for s in clean:
        if not merged: merged.append(s)
        else:
            prev = merged[-1]
            if (s["end"] - s["start"]) < merge_threshold or (s["start"] - prev["start"] < merge_threshold):
                prev["end"] = max(prev["end"], s["end"])
                if not prev.get("callout_text") and s.get("callout_text"):
                    prev["callout_text"] = s["callout_text"]; prev["callout_type"] = s["callout_type"]
            else: merged.append(s)
    clean = merged
    clean[0]["start"] = 0.0
    for i in range(len(clean) - 1): clean[i]["end"] = clean[i + 1]["start"]
    clean[-1]["end"] = batch_duration

    split_threshold = max(max_s * 1.3, 15.0)
    final = []
    for s in clean:
        dur = s["end"] - s["start"]
        if dur > split_threshold:
            mid = s["start"] + dur / 2.0
            final.append({**s, "end": mid})
            final.append({**s, "start": mid, "title": f"{s['title']} (TIẾP)",
                "callout_type": "sticker", "callout_text": "!", "text_boxes": []})
        else: final.append(s)

    if camera_mode == "random":
        for s in final: s["camera_motion"] = random.choice(list(valid_motions))
    elif camera_mode.startswith("fixed:"):
        fixed = camera_mode.split(":", 1)[1].strip()
        if fixed in valid_motions:
            for s in final: s["camera_motion"] = fixed
    return final

# ============================================================
# IMAGE PROVIDERS
# ============================================================
def _build_full_prompt(prompt, chars=None, char_lock=None, style_mode="comic",
                       language="vi", title="", callout_text=""):
    chars = chars or {}
    native_text = (language == "en" and style_mode == "comic")

    if style_mode == "horror":
        safe = horror_sanitize(prompt)
        style = """Dramatic dark illustration, cinematic horror atmosphere, deep shadows.
Rich moody backgrounds, dramatic lighting.
Absolutely NO text, letters, numbers, captions.
Wide 16:9 cinematic composition.
CHARACTER GENDER: male = MALE, female = FEMALE."""
    elif native_text:
        safe = prompt
        style = """COLORED CARTOON ILLUSTRATION with natural warm earth tones.
Color palette: brown, tan, orange, yellow, beige, warm gray, olive, deep green.
Soft textures, warm firelight glow, soft shadows, cinematic lighting.
NOT pure white background. Outdoor: natural sky. Indoor: warm cave/room tones.
Thick black outlines, hand-drawn doodle style.
EXPRESSIVE CARTOON CHARACTERS with clear emotions.

⭐ TITLE STYLE (CRITICAL):
- STICKER-STYLE: bold letters with THICK contrasting outline.
- Example: white letters with thick black outline, OR yellow with thick black outline.
- Title placed DIRECTLY on illustration, floating over the scene naturally.
- ABSOLUTELY NO rectangular band, NO banner, NO black bar, NO frame behind title.
- Think "comic book title sticker" NOT "YouTube banner".
- FORBIDDEN: any rectangle, band, bar, or frame behind the title text.

RENDER TEXT VISIBLY: title, speech bubbles, labels. Text MUST be correctly spelled.
Wide 16:9 cinematic composition.
CHARACTER GENDER: male = MALE, female = FEMALE."""
    else:
        safe = sanitize_prompt_text(prompt)
        style = """Authentic 2D comic doodle art style, thick black ink contour outlines.
Pure solid flat white background.
Vivid expressive cartoon character.
Selective vibrant spot colors on key elements.
Absolutely NO text, letters, numbers, captions, or empty speech balloons.
Wide 16:9 cinematic composition.
CHARACTER GENDER: male = MALE, female = FEMALE."""

    if char_lock: safe = enforce_character_lock(safe, char_lock)
    if char_lock and char_lock.get("__global__"):
        safe += "\n\nIMPORTANT: The MAIN CHARACTER MUST be visible and identical."

    if native_text:
        text_instructions = []
        if title:
            text_instructions.append(
                f'Title "{title}" as sticker-style text: bold letters with thick contrasting outline '
                f'(white or yellow fill, thick black outline), integrated into scene naturally, '
                f'NO rectangular banner, NO black background bar, NO frame behind text'
            )
        if callout_text and callout_text.strip():
            text_instructions.append(f'Speech bubble with text "{callout_text}"')
        if text_instructions:
            safe += "\n\nTEXT TO RENDER: " + ". ".join(text_instructions) + "."
            safe += "\nMANDATORY: Do NOT draw any rectangular shape, band, banner, or frame behind the title."

    return f"{safe}.\n\nSTYLE CONSTRAINTS:\n{style}"

def _validate(data, name):
    if not data or len(data) < 500: raise RuntimeError(f"{name}: dữ liệu quá nhỏ")
    if not (data[:3] == b'\xff\xd8\xff' or data[:8] == b'\x89PNG\r\n\x1a\n' or data[:4] == b'RIFF'):
        pv = data[:150].decode("utf-8", errors="ignore").lower()
        if "<html" in pv or "<!doctype" in pv: raise RuntimeError(f"{name}: HTML")
        raise RuntimeError(f"{name}: không phải ảnh")
    return data

def agnes_image_request(prompt, api_key, timeout=45, chars=None, char_lock=None, seed=None,
                        style_mode="comic", language="vi", title="", callout_text=""):
    api_key = (api_key or "").strip()
    if not api_key: raise RuntimeError("Agnes: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": AGNES_MODEL,
               "prompt": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text),
               "size": "1280x720", "extra_body": {"response_format": "b64_json"}}
    r = requests.post(AGNES_API_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429: raise RuntimeError("Agnes: rate limit")
    if r.status_code == 401: raise RuntimeError("Agnes: key sai")
    if r.status_code >= 400: raise RuntimeError(f"Agnes HTTP {r.status_code}")
    data = r.json(); item = data.get("data", [{}])[0]
    if item.get("b64_json"): return _validate(base64.b64decode(item["b64_json"]), "Agnes")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate(img.content, "Agnes")
    raise RuntimeError("Agnes: no image")

def cloudflare_image_request(prompt, account_id, api_token, timeout=45, steps=4, chars=None,
                              char_lock=None, seed=None, style_mode="comic",
                              language="vi", title="", callout_text=""):
    account_id = (account_id or "").strip(); api_token = (api_token or "").strip()
    if not account_id or not api_token: raise RuntimeError("Cloudflare: thiếu thông tin")
    url = f"{CLOUDFLARE_BASE}{account_id}/ai/run/{CLOUDFLARE_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {"prompt": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text),
               "steps": steps}
    if seed is not None: payload["seed"] = int(seed)
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429: raise RuntimeError("Cloudflare: hết quota")
    if r.status_code >= 400: raise RuntimeError(f"Cloudflare HTTP {r.status_code}")
    data = r.json()
    if not data.get("success", True): raise RuntimeError("Cloudflare fail")
    b64 = data.get("result", {}).get("image")
    if not b64: raise RuntimeError("Cloudflare: no image")
    return _validate(base64.b64decode(b64), "Cloudflare")

def hf_image_request(prompt, token, timeout=45, chars=None, char_lock=None, seed=None,
                     style_mode="comic", language="vi", title="", callout_text=""):
    token = (token or "").strip()
    if not token: raise RuntimeError("HF: chưa có token")
    url = f"{HF_API_URL}{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text)}
    if seed is not None: payload["parameters"] = {"seed": int(seed)}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 503: raise RuntimeError("HF: model loading")
    if r.status_code == 429: raise RuntimeError("HF: rate limit")
    if r.status_code >= 400: raise RuntimeError(f"HF HTTP {r.status_code}")
    return _validate(r.content, "HF")

def freetheai_image_request(prompt, api_key, timeout=45, chars=None, char_lock=None, seed=None,
                             style_mode="comic", language="vi", title="", callout_text=""):
    api_key = (api_key or "").strip()
    if not api_key: raise RuntimeError("FreeTheAi: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "flux",
               "prompt": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text),
               "n": 1, "size": "1280x720"}
    if seed is not None: payload["seed"] = int(seed)
    r = requests.post(FREETHEAI_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429: raise RuntimeError("FreeTheAi: rate limit")
    if r.status_code >= 400: raise RuntimeError(f"FreeTheAi HTTP {r.status_code}")
    data = r.json(); item = data.get("data", [{}])[0]
    if item.get("b64_json"): return _validate(base64.b64decode(item["b64_json"]), "FreeTheAi")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate(img.content, "FreeTheAi")
    raise RuntimeError("FreeTheAi: no image")

def together_image_request(prompt, api_key, timeout=45, chars=None, char_lock=None, seed=None,
                            style_mode="comic", language="vi", title="", callout_text=""):
    api_key = (api_key or "").strip()
    if not api_key: raise RuntimeError("Together: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": TOGETHER_MODEL,
               "prompt": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text),
               "width": WIDTH, "height": HEIGHT, "steps": 4, "n": 1, "response_format": "b64_json"}
    r = requests.post(TOGETHER_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429: raise RuntimeError("Together: rate limit")
    if r.status_code >= 400: raise RuntimeError(f"Together HTTP {r.status_code}")
    data = r.json(); item = data.get("data", [{}])[0]
    if item.get("b64_json"): return _validate(base64.b64decode(item["b64_json"]), "Together")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate(img.content, "Together")
    raise RuntimeError("Together: no image")

def nexa_image_request(prompt, api_key, timeout=45, chars=None, char_lock=None, seed=None,
                        style_mode="comic", language="vi", title="", callout_text=""):
    api_key = (api_key or "").strip()
    if not api_key: raise RuntimeError("NexaAPI: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": NEXA_MODEL,
               "prompt": _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text),
               "width": WIDTH, "height": HEIGHT, "n": 1}
    if seed is not None: payload["seed"] = int(seed)
    r = requests.post(NEXA_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429: raise RuntimeError("NexaAPI: rate limit")
    if r.status_code >= 400: raise RuntimeError(f"NexaAPI HTTP {r.status_code}")
    data = r.json(); item = data.get("data", [{}])[0]
    if item.get("b64_json"): return _validate(base64.b64decode(item["b64_json"]), "NexaAPI")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate(img.content, "NexaAPI")
    raise RuntimeError("NexaAPI: no image")

def pollinations_image_request(prompt, api_key, model="flux-pro", timeout=45, seed=None, chars=None,
                                char_lock=None, style_mode="comic", language="vi", title="", callout_text=""):
    api_key = (api_key or "").strip()
    if not api_key: raise RuntimeError("Pollinations: chưa có key")
    encoded = requests.utils.quote(
        _build_full_prompt(prompt, chars, char_lock, style_mode, language, title, callout_text), safe="")
    url = f"{POLLINATIONS_BASE}{encoded}"
    params = {"width": WIDTH, "height": HEIGHT, "model": model,
              "nologo": "true", "enhance": "true", "safe": "false"}
    if seed is not None: params["seed"] = int(seed)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=True)
    if r.status_code == 401: raise RuntimeError("Pollinations 401")
    if r.status_code == 402: raise RuntimeError("Pollinations 402")
    if r.status_code == 429: raise RuntimeError("Pollinations 429")
    if r.status_code >= 400: raise RuntimeError(f"Pollinations HTTP {r.status_code}")
    return _validate(r.content, "Pollinations")

def build_provider_list(cf_account, cf_token, hf_token, freetheai_key, together_key, nexa_key, agnes_key,
                         pollinations_key="", pollinations_model="flux-pro", flux_steps=4,
                         chars=None, char_lock=None, seed=None, style_mode="comic"):
    chars = chars or {}; char_lock = char_lock or {}
    kw = {"chars": chars, "char_lock": char_lock, "seed": seed, "style_mode": style_mode}
    providers = []
    if cf_account and cf_token and cf_account.strip() and cf_token.strip():
        providers.append({"name": "Cloudflare", "fn": cloudflare_image_request,
            "args": [cf_account.strip(), cf_token.strip()], "kwargs": {"steps": flux_steps, **kw}})
    if agnes_key and agnes_key.strip():
        providers.append({"name": "Agnes AI", "fn": agnes_image_request, "args": [agnes_key.strip()], "kwargs": kw})
    if together_key and together_key.strip():
        providers.append({"name": "Together AI", "fn": together_image_request, "args": [together_key.strip()], "kwargs": kw})
    if freetheai_key and freetheai_key.strip():
        providers.append({"name": "FreeTheAi", "fn": freetheai_image_request, "args": [freetheai_key.strip()], "kwargs": kw})
    if hf_token and hf_token.strip():
        providers.append({"name": "Hugging Face", "fn": hf_image_request, "args": [hf_token.strip()], "kwargs": kw})
    if nexa_key and nexa_key.strip():
        providers.append({"name": "NexaAPI", "fn": nexa_image_request, "args": [nexa_key.strip()], "kwargs": kw})
    if pollinations_key and pollinations_key.strip():
        providers.append({"name": f"Pollinations ({pollinations_model})", "fn": pollinations_image_request,
            "args": [pollinations_key.strip(), pollinations_model], "kwargs": kw})
    return providers

def save_image(data, output_path):
    if not data: raise RuntimeError("Dữ liệu rỗng")
    Path(output_path).write_bytes(data)
    try:
        with Image.open(output_path) as im: im.verify()
        with Image.open(output_path) as im:
            im = im.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            im.save(output_path, "JPEG", quality=95)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Ảnh invalid: {e}")

# ============================================================
# OVERLAY
# ============================================================
def draw_arrow(draw, s, e, color, w=5):
    draw.line([s, e], fill=color, width=w)
    ang = math.atan2(e[1] - s[1], e[0] - s[0])
    al = 20; aa = math.pi / 5
    p1 = (e[0] - al * math.cos(ang - aa), e[1] - al * math.sin(ang - aa))
    p2 = (e[0] - al * math.cos(ang + aa), e[1] - al * math.sin(ang + aa))
    draw.polygon([e, p1, p2], fill=color)

def draw_text_shadow(draw, xy, text, font, fill, sw=0, sf=None, shadow=True):
    x, y = xy
    if shadow: draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 80))
    if sw > 0 and sf: draw.text((x, y), text, font=font, fill=fill, stroke_width=sw, stroke_fill=sf)
    else: draw.text((x, y), text, font=font, fill=fill)

def draw_rich_text_box(img, draw, tb, enable_shadow=True):
    try:
        x = int(tb["x"] * WIDTH); y = int(tb["y"] * HEIGHT)
        text = tb["text"]; color = COLOR_MAP.get(tb["color"], "#212121")
        stl = tb["style"]
        size_key = tb["size"]
        if len(text) > 20 and size_key in ("large", "huge"): size_key = "medium"
        if len(text) > 30 and size_key == "medium": size_key = "small"
        f_size = SIZE_MAP.get(size_key, 30)
        f = font_for(f_size, bold=True)
        bbox = draw.textbbox((0, 0), text, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_end_x = x + tw; text_end_y = y + th
        if text_end_x > WIDTH * 0.30 and x < WIDTH * 0.70:
            if text_end_y > HEIGHT * 0.30 and y < HEIGHT * 0.75:
                return
        if x + tw + 20 > WIDTH: x = max(10, WIDTH - tw - 20)
        if y + th + 20 > HEIGHT: y = max(TITLE_BAND_H + 10, HEIGHT - th - 20)
        if x < 10: x = 10
        if y < TITLE_BAND_H + 10: y = TITLE_BAND_H + 10
        if stl == "highlighted":
            px, py = 16, 10
            rect = [x - px, y - py, x + tw + px, y + th + py + 6]
            if enable_shadow: draw.rounded_rectangle([rect[0]+3, rect[1]+3, rect[2]+3, rect[3]+3], radius=10, fill=(0,0,0,60))
            draw.rounded_rectangle(rect, radius=10, fill=color, outline="white", width=3)
            draw.text((x, y), text, fill="white", font=f)
        elif stl == "outlined":
            draw_text_shadow(draw, (x, y), text, f, fill=color, sw=3, sf="white", shadow=enable_shadow)
        else:
            px, py = 10, 6
            rect = [x - px, y - py, x + tw + px, y + th + py + 4]
            bg = Image.new("RGBA", (rect[2]-rect[0], rect[3]-rect[1]), (255,255,255,200))
            img.paste(bg, (rect[0], rect[1]), bg)
            draw.text((x, y), text, fill=color, font=f)
    except Exception: pass

def smart_filter_tbs(tbs, callout_text, callout_type, callout_side):
    filtered = []; min_dist = 0.20
    for tb in tbs:
        x, y = tb["x"], tb["y"]
        if y < 0.25: continue
        in_corners = (
            (x < 0.30 and y < 0.40) or (x > 0.65 and y < 0.40) or
            (x < 0.30 and y > 0.72) or (x > 0.65 and y > 0.72)
        )
        if not in_corners: continue
        if callout_text and callout_type != "none":
            if callout_side == "left":
                if x < 0.45 and 0.35 < y < 0.55: continue
            else:
                if x > 0.55 and 0.35 < y < 0.55: continue
        close = False
        for ex in filtered:
            if abs(ex["x"] - x) < min_dist and abs(ex["y"] - y) < min_dist:
                close = True; break
        if not close: filtered.append(tb)
        if len(filtered) >= 3: break
    return filtered

def add_comic_overlays(image_path, title, callout_type, callout_text, callout_side, output_path,
                        text_boxes=None, enable_arrows=True, enable_shadow=True,
                        style_mode="comic", language="vi"):
    img = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))

    if language == "en" and style_mode == "comic":
        img.save(output_path, quality=95)
        return

    draw = ImageDraw.Draw(img)
    is_horror = (style_mode == "horror")
    title_color = "#1a0000" if is_horror else "#111111"
    underline_color = "#8b0000" if is_horror else "#d32f2f"
    bubble_text_color = "#6a0000" if is_horror else "#1b5e20"
    thought_text_color = "#1a0033" if is_horror else "#0d47a1"

    if title:
        band_bg = "#0a0a0a" if is_horror else "white"
        draw.rectangle([0, 0, WIDTH, TITLE_BAND_H], fill=band_bg)
        f_size = 38; f_title = font_for(f_size, bold=True)
        while f_size > 20:
            box = draw.textbbox((0, 0), title, font=f_title)
            if box[2] - box[0] <= 900: break
            f_size -= 2; f_title = font_for(f_size, bold=True)
        box = draw.textbbox((0, 0), title, font=f_title)
        tw = box[2] - box[0]
        text_x = (WIDTH - tw) / 2; text_y = 22
        draw.text((text_x, text_y), title, fill=title_color, font=f_title)
        underline_y = text_y + box[3] + 10
        draw.line([(text_x, underline_y), (text_x + tw, underline_y)], fill=underline_color, width=4)

    if callout_text and callout_type != "none":
        f_text = font_for(26, bold=True)
        bb = draw.textbbox((0, 0), callout_text, font=f_text)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        cx, cy = (int(WIDTH * 0.28), int(HEIGHT * 0.45)) if callout_side == "left" else (int(WIDTH * 0.74), int(HEIGHT * 0.42))
        if callout_type == "speech":
            px, py = 20, 14
            rect = [cx - bw // 2 - px, cy - bh // 2 - py, cx + bw // 2 + px, cy + bh // 2 + py]
            if enable_shadow: draw.rounded_rectangle([rect[0]+4, rect[1]+4, rect[2]+4, rect[3]+4], radius=16, fill=(0,0,0,50))
            draw.rounded_rectangle(rect, radius=16, fill="white", outline="black", width=4)
            tail = (cx - 20, cy + bh // 2 + py + 22)
            draw.polygon([(cx - 34, cy + bh // 2 + py - 2), (cx - 8, cy + bh // 2 + py - 2), tail], fill="white", outline="black")
            draw.line([(cx - 32, cy + bh // 2 + py), (cx - 10, cy + bh // 2 + py)], fill="white", width=5)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill=bubble_text_color, font=f_text)
        elif callout_type == "thought":
            px, py = 24, 16
            rect = [cx - bw // 2 - px, cy - bh // 2 - py, cx + bw // 2 + px, cy + bh // 2 + py]
            if enable_shadow: draw.rounded_rectangle([rect[0]+4, rect[1]+4, rect[2]+4, rect[3]+4], radius=26, fill=(0,0,0,50))
            draw.rounded_rectangle(rect, radius=26, fill="white", outline="black", width=3)
            draw.ellipse([cx - 22, cy + bh // 2 + py + 6, cx - 12, cy + bh // 2 + py + 16], fill="white", outline="black", width=3)
            draw.ellipse([cx - 30, cy + bh // 2 + py + 19, cx - 24, cy + bh // 2 + py + 25], fill="white", outline="black", width=2)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill=thought_text_color, font=f_text)
        else:
            px, py = 18, 11
            bx, by = int(WIDTH * 0.75), int(HEIGHT * 0.88)
            r = [bx - bw // 2 - px, by - bh // 2 - py, bx + bw // 2 + px, by + bh // 2 + py]
            if enable_shadow: draw.rounded_rectangle([r[0]+3, r[1]+3, r[2]+3, r[3]+3], radius=10, fill=(0,0,0,60))
            outline = "#5a0000" if is_horror else "#b71c1c"
            text_col = "#5a0000" if is_horror else "#b71c1c"
            draw.rounded_rectangle(r, radius=10, fill="white", outline=outline, width=4)
            draw.text((bx - bw // 2, by - bh // 2 - 2), callout_text, fill=text_col, font=f_text)

    if text_boxes:
        filtered = smart_filter_tbs(text_boxes, callout_text, callout_type, callout_side)
        if enable_arrows:
            for tb in filtered:
                arr = tb.get("arrow")
                if not arr: continue
                try:
                    f = font_for(SIZE_MAP.get(tb["size"], 30), bold=True)
                    bbox = draw.textbbox((0, 0), tb["text"], font=f)
                    tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
                    sx = int(tb["x"] * WIDTH) + tw // 2
                    sy = int(tb["y"] * HEIGHT) + th // 2
                    ex = int(arr["to_x"] * WIDTH); ey = int(arr["to_y"] * HEIGHT)
                    if abs(ex - sx) < 50 and abs(ey - sy) < 50: continue
                    if ex < 60 or ex > WIDTH - 60 or ey < 150 or ey > HEIGHT - 60: continue
                    if math.hypot(ex - sx, ey - sy) > int(WIDTH * 0.35): continue
                    color = COLOR_MAP.get(tb["color"], "#212121")
                    draw_arrow(draw, (sx, sy), (ex, ey), color, w=5)
                except Exception: pass
        for tb in filtered: draw_rich_text_box(img, draw, tb, enable_shadow)

    img.save(output_path, quality=95)

# ============================================================
# HAND + TRAJECTORY
# ============================================================
def fallback_hand():
    S = 320
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.line([(40, 40), (140, 140)], fill=(30,30,30,255), width=10)
    d.polygon([(30,30),(50,40),(40,50)], fill=(220,50,50,255))
    d.ellipse((110,110,260,260), fill=(235,205,175,255), outline=(30,30,30,255), width=4)
    d.rounded_rectangle((100,130,180,220), 20, fill=(235,205,175,255), outline=(30,30,30,255), width=4)
    return im

def load_hand(hand_path, tw=320):
    p = None
    for c in [hand_path, Path("hand.png"), Path("assets/hand.png")]:
        if c and Path(c).exists(): p = Path(c); break
    ph = Image.open(p).convert("RGBA") if p else fallback_hand()
    w, h = ph.size; ph = ph.resize((tw, int(h * tw / w)), Image.Resampling.LANCZOS)
    hn = np.array(ph)
    bgr = cv2.cvtColor(hn[:, :, :3], cv2.COLOR_RGB2BGR); alpha = hn[:, :, 3]
    alpha[:6, :] = 0; alpha[-6:, :] = 0; alpha[:, :6] = 0; alpha[:, -6:] = 0
    alpha[alpha < 110] = 0
    nl, lb, st, _ = cv2.connectedComponentsWithStats((alpha > 50).astype(np.uint8))
    if nl > 1: alpha[lb != (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.erode(alpha, k, iterations=1)
    alpha = np.minimum(alpha, cv2.GaussianBlur(alpha, (3, 3), 0))
    ys, xs = np.where(alpha > 120)
    tx, ty = (int(xs[np.argmin(xs + ys * 1.15)]), int(ys[np.argmin(xs + ys * 1.15)])) if len(xs) > 0 else (0, 0)
    return bgr, alpha, tx, ty

def sort_nn(contours, start=(100, 150)):
    def cc(c):
        M = cv2.moments(c)
        if M["m00"] > 0: return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        x, y, w, h = cv2.boundingRect(c); return (x + w//2, y + h//2)
    valid = [c for c in contours if cv2.arcLength(c, False) > 10]
    out = []
    if valid:
        cur = start; rem = valid[:]
        while rem:
            bi = 0; bd = float("inf")
            for i, c in enumerate(rem):
                pt = cc(c); d = (pt[0]-cur[0])**2 + (pt[1]-cur[1])**2
                if d < bd: bd = d; bi = i
            ch = rem.pop(bi); out.append(ch); cur = cc(ch)
    return out

def extract_cont_traj(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return [(WIDTH//2, HEIGHT//2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    tm = np.zeros_like(binary); tm[:TITLE_BAND_H, :] = binary[:TITLE_BAND_H, :]
    bm = np.zeros_like(binary); bm[TITLE_BAND_H:, :] = binary[TITLE_BAND_H:, :]
    tc, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bc, _ = cv2.findContours(bm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ts = sorted(tc, key=lambda c: cv2.boundingRect(c)[0])
    bs = sort_nn(bc, start=(150, 200))
    traj = []
    for c in ts + bs:
        for p in c.reshape(-1, 2)[::3]: traj.append((int(p[0]), int(p[1])))
    if len(traj) < 40: traj = [(x, y) for y in range(120, 680, 45) for x in range(80, 1200, 25)]
    return traj

def extract_stag_traj(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return [[(WIDTH//2, HEIGHT//2)]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    mt = np.zeros_like(binary); mt[:TITLE_BAND_H, :] = binary[:TITLE_BAND_H, :]
    ml = np.zeros_like(binary); ml[TITLE_BAND_H:HEIGHT, :int(WIDTH*0.48)] = binary[TITLE_BAND_H:HEIGHT, :int(WIDTH*0.48)]
    mr = np.zeros_like(binary); mr[TITLE_BAND_H:int(HEIGHT*0.72), int(WIDTH*0.48):] = binary[TITLE_BAND_H:int(HEIGHT*0.72), int(WIDTH*0.48):]
    mb = np.zeros_like(binary); mb[int(HEIGHT*0.72):, int(WIDTH*0.48):] = binary[int(HEIGHT*0.72):, int(WIDTH*0.48):]
    def tp(cs): return [(int(p[0]), int(p[1])) for c in cs for p in c.reshape(-1, 2)[::3]]
    tc, _ = cv2.findContours(mt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lc, _ = cv2.findContours(ml, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rc, _ = cv2.findContours(mr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    bc, _ = cv2.findContours(mb, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    zones = [tp(sorted(tc, key=lambda c: cv2.boundingRect(c)[0])), tp(sort_nn(lc, (150,250))),
             tp(sort_nn(rc, (int(WIDTH*0.7), 250))), tp(sort_nn(bc, (int(WIDTH*0.75), int(HEIGHT*0.85))))]
    return [z for z in zones if len(z) > 10]

def paste_hand(f, hb, ha, x, y):
    fh, fw = f.shape[:2]; hh, hw = hb.shape[:2]
    x1, y1 = max(0, x), max(0, y); x2, y2 = min(fw, x+hw), min(fh, y+hh)
    if x1 >= x2 or y1 >= y2: return
    hx1, hy1 = x1-x, y1-y; hx2, hy2 = hx1+(x2-x1), hy1+(y2-y1)
    sh = hb[hy1:hy2, hx1:hx2]; sa = (ha[hy1:hy2, hx1:hx2].astype(np.float32)/255.0)[:, :, None]
    roi = f[y1:y2, x1:x2].astype(np.float32)
    bl = sh.astype(np.float32) * sa + roi * (1.0 - sa)
    f[y1:y2, x1:x2] = np.clip(bl, 0, 255).astype(np.uint8)

def ease(t): return 0.5 * (1.0 - math.cos(math.pi * t))

COMIC_MOTIONS = {
    "zoom_in_center": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.22, WIDTH*0.5, HEIGHT*0.5)],
    "zoom_out_center": [(0.0, 1.22, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.00, WIDTH*0.5, HEIGHT*0.5)],
    "pan_left_to_right": [(0.0, 1.12, WIDTH*0.35, HEIGHT*0.5), (1.0, 1.12, WIDTH*0.65, HEIGHT*0.5)],
    "pan_right_to_left": [(0.0, 1.12, WIDTH*0.65, HEIGHT*0.5), (1.0, 1.12, WIDTH*0.35, HEIGHT*0.5)],
    "zoom_in_top_left": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.24, WIDTH*0.30, HEIGHT*0.35)],
    "zoom_in_bottom_right": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.24, WIDTH*0.70, HEIGHT*0.65)],
    "ken_burns_slow": [(0.0, 1.05, WIDTH*0.45, HEIGHT*0.48), (1.0, 1.18, WIDTH*0.55, HEIGHT*0.52)],
    "static": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.00, WIDTH*0.5, HEIGHT*0.5)],
}

EN_COMIC_MOTIONS = {
    "zoom_in_center": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.12, WIDTH*0.5, HEIGHT*0.5)],
    "zoom_out_center": [(0.0, 1.12, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.00, WIDTH*0.5, HEIGHT*0.5)],
    "pan_left_to_right": [(0.0, 1.08, WIDTH*0.40, HEIGHT*0.5), (1.0, 1.08, WIDTH*0.60, HEIGHT*0.5)],
    "pan_right_to_left": [(0.0, 1.08, WIDTH*0.60, HEIGHT*0.5), (1.0, 1.08, WIDTH*0.40, HEIGHT*0.5)],
    "zoom_in_top_left": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.12, WIDTH*0.35, HEIGHT*0.40)],
    "zoom_in_bottom_right": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.12, WIDTH*0.65, HEIGHT*0.60)],
    "ken_burns_slow": [(0.0, 1.02, WIDTH*0.47, HEIGHT*0.49), (1.0, 1.10, WIDTH*0.53, HEIGHT*0.51)],
    "static": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.00, WIDTH*0.5, HEIGHT*0.5)],
}

HORROR_MOTIONS = {
    "slow_zoom_in": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.15, WIDTH*0.5, HEIGHT*0.5)],
    "slow_zoom_out": [(0.0, 1.15, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.00, WIDTH*0.5, HEIGHT*0.5)],
    "creepy_pan_left": [(0.0, 1.08, WIDTH*0.42, HEIGHT*0.5), (1.0, 1.08, WIDTH*0.58, HEIGHT*0.5)],
    "creepy_pan_right": [(0.0, 1.08, WIDTH*0.58, HEIGHT*0.5), (1.0, 1.08, WIDTH*0.42, HEIGHT*0.5)],
    "dramatic_zoom_face": [(0.0, 1.00, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.35, WIDTH*0.5, HEIGHT*0.40)],
    "static_dread": [(0.0, 1.05, WIDTH*0.5, HEIGHT*0.5), (1.0, 1.05, WIDTH*0.5, HEIGHT*0.5)],
}

def get_motion_kfs(motion, style_mode="comic", language="vi"):
    if style_mode == "horror": src = HORROR_MOTIONS
    elif language == "en": src = EN_COMIC_MOTIONS
    else: src = COMIC_MOTIONS
    return src.get(motion, list(src.values())[0])

def interp_motion(kfs, p):
    if p <= kfs[0][0]: _, s, cx, cy = kfs[0]; return s, cx, cy
    if p >= kfs[-1][0]: _, s, cx, cy = kfs[-1]; return s, cx, cy
    for i in range(len(kfs)-1):
        k0, k1 = kfs[i], kfs[i+1]
        if k0[0] <= p <= k1[0]:
            sp = k1[0] - k0[0]
            lc = (p - k0[0]) / sp if sp > 0 else 1.0
            e = ease(lc)
            return (k0[1] + (k1[1]-k0[1])*e, k0[2] + (k1[2]-k0[2])*e, k0[3] + (k1[3]-k0[3])*e)
    _, s, cx, cy = kfs[-1]; return s, cx, cy

# ============================================================
# RENDER 4 STYLES — V10.3: EN skip draw
# ============================================================
def render_kttv(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                style_mode="comic", language="vi"):
    tf = max(1, round(duration * FPS))
    use_full_frame = (language == "en" and style_mode == "comic")
    # V10.3: English bỏ hiệu ứng vẽ — hiện ảnh ngay
    if use_full_frame:
        dd = 0.001; df = 0; rf = 0
    else:
        dd = max(1.5, min(duration - 0.8, duration * DRAW_DURATION_RATIO))
        df = int(dd * FPS); rf = int(0.35 * FPS)

    of = cv2.imread(str(image_path))
    if of is None: raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    of = cv2.resize(of, (WIDTH, HEIGHT))

    if use_full_frame:
        tb = None; cb = of
    else:
        tb, cb = split_title_band(of)

    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)
    if use_full_frame:
        rm[:, :] = 255  # Full reveal ngay từ đầu

    zt = extract_stag_traj(image_path)
    ap_full = [p for z in zt for p in z] or [(WIDTH//2, HEIGHT//2)]
    ap = [(px, py) for (px, py) in ap_full] if use_full_frame else traj_to_content(ap_full)
    ph = split_traj_phases(ap)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lt = ap[0] if ap else (WIDTH//2, ch_use//2)
    kfs = get_motion_kfs(motion, style_mode, language)

    for fi in range(tf):
        hv = False; hx = hy = 0
        if fi < df and not use_full_frame:
            if fi < pf[0]: cp=0; lp=fi/max(1,pf[0])
            elif fi < pf[1]: cp=1; lp=(fi-pf[0])/max(1,pf[1]-pf[0])
            else: cp=2; lp=(fi-pf[1])/max(1,pf[2]-pf[1])
            for pi in range(cp):
                for pt in ph[pi]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
            cps = ph[cp]
            if cps:
                cnt = max(1, min(int(lp*len(cps)), len(cps)))
                for pt in cps[:cnt]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
                tg = cps[cnt-1]
            else: tg = lt
            hx = tg[0]+int(1.2*math.sin(fi*1.8)); hy = tg[1]+int(1.2*math.cos(fi*1.8))
            lt = (hx, hy); hv = True
        elif fi < df+rf and not use_full_frame:
            rm[:, :] = 255; pr = (fi-df)/max(1,rf)
            hx = int(lt[0]+(WIDTH+180-lt[0])*pr); hy = int(lt[1]+(ch_use+180-lt[1])*pr); hv = True
        elif not use_full_frame:
            rm[:, :] = 255

        a = (cv2.GaussianBlur(rm, (13, 13), 0).astype(np.float32)/255.0)[:, :, None]
        fc = (cb*a + wc*(1.0-a)).astype(np.uint8)
        if hv: paste_hand(fc, hb, ha, hx-tx, hy-ty)

        # V10.3: EN mode → camera motion chạy từ frame 0
        if use_full_frame or fi >= df+rf:
            op = fi/max(1, tf-1) if use_full_frame else (fi-df-rf)/max(1, tf-df-rf)
            st_, cx_, cy_ = interp_motion(kfs, op)
            cyc = (cy_/HEIGHT)*ch_use
            bl = ease(min(1.0, op*1.8)) if not use_full_frame else ease(op)
            sc = 1.0 + (st_-1.0)*bl
            cu = WIDTH*0.5 + (cx_-WIDTH*0.5)*bl
            cyu = ch_use*0.5 + (cyc-ch_use*0.5)*bl
        else:
            sc, cu, cyu = 1.0, WIDTH*0.5, ch_use*0.5

        if use_full_frame:
            fo = crop_full_frame(fc, sc, cu, cyu)
        else:
            fo = compose_frame(tb, crop_content_motion(fc, sc, cu, cyu))
        proc.stdin.write(fo.tobytes())
    proc.stdin.close(); proc.wait()
    if proc.returncode != 0: raise RuntimeError("FFmpeg fail (style 1)")

def render_hybrid(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                  style_mode="comic", language="vi"):
    # V10.3: Với English, dùng render_kttv thay vì hybrid vẽ tay
    if language == "en" and style_mode == "comic":
        return render_kttv(image_path, duration, output_path, hand_path, motion, style_mode, language)

    tf = max(1, round(duration * FPS))
    dd = max(1.5, min(duration - 0.8, duration * DRAW_DURATION_RATIO))
    df = int(dd*FPS); rf = int(0.35*FPS)
    of = cv2.imread(str(image_path))
    if of is None: raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    of = cv2.resize(of, (WIDTH, HEIGHT))
    tb, cb = split_title_band(of)
    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)
    tr_full = extract_cont_traj(image_path); tr = traj_to_content(tr_full)
    ph = split_traj_phases(tr)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lt = tr[0] if tr else (WIDTH//2, ch_use//2)
    scx, scy = float(lt[0]), float(lt[1])
    kfs = get_motion_kfs(motion, style_mode, language)
    for fi in range(tf):
        hv = False; hx = hy = 0
        if fi < df:
            if fi < pf[0]: cp=0; lp=fi/max(1,pf[0])
            elif fi < pf[1]: cp=1; lp=(fi-pf[0])/max(1,pf[1]-pf[0])
            else: cp=2; lp=(fi-pf[1])/max(1,pf[2]-pf[1])
            for pi in range(cp):
                for pt in ph[pi]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
            cps = ph[cp]
            if cps:
                cnt = max(1, min(int(lp*len(cps)), len(cps)))
                for pt in cps[:cnt]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
                tg = cps[cnt-1]
            else: tg = lt
            hx = tg[0]+int(1.2*math.sin(fi*1.8)); hy = tg[1]+int(1.2*math.cos(fi*1.8))
            lt = (hx, hy); hv = True
        elif fi < df+rf:
            rm[:, :] = 255; pr = (fi-df)/max(1,rf)
            hx = int(lt[0]+(WIDTH+180-lt[0])*pr); hy = int(lt[1]+(ch_use+180-lt[1])*pr); hv = True
        else: rm[:, :] = 255
        a = (cv2.GaussianBlur(rm, (13, 13), 0).astype(np.float32)/255.0)[:, :, None]
        fc = (cb*a + wc*(1.0-a)).astype(np.uint8)
        if hv: paste_hand(fc, hb, ha, hx-tx, hy-ty)
        if fi < df:
            sc = 1.0
            scx = scx*0.95 + hx*0.05; scy = scy*0.95 + hy*0.05
            cu, cyu = scx, scy
        elif fi < df+rf: sc, cu, cyu = 1.0, scx, scy
        else:
            op = (fi-df-rf)/max(1, tf-df-rf)
            st_, cx_, cy_ = interp_motion(kfs, op); cyc = (cy_/HEIGHT)*ch_use
            bl = ease(min(1.0, op*1.8)); sc = 1.0 + (st_-1.0)*bl
            cu = scx + (cx_-scx)*bl; cyu = scy + (cyc-scy)*bl
        cw = int(WIDTH/sc); chh = int(ch_use/sc)
        ccx = max(cw//2, min(WIDTH-cw//2, int(cu))); ccy = max(chh//2, min(ch_use-chh//2, int(cyu)))
        fo = compose_frame(tb, crop_content_motion(fc, sc, ccx, ccy))
        proc.stdin.write(fo.tobytes())
    proc.stdin.close(); proc.wait()

def render_pure(image_path, duration, output_path, motion="zoom_in_center",
                style_mode="comic", language="vi"):
    tf = max(1, round(duration * FPS))
    of = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    use_full_frame = (language == "en" and style_mode == "comic")
    if use_full_frame:
        tb = None; cb = of
    else:
        tb, cb = split_title_band(of)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    kfs = get_motion_kfs(motion, style_mode, language)
    ch_use = cb.shape[0]
    for fi in range(tf):
        p = fi / max(1, tf-1)
        s, cx, cyf = interp_motion(kfs, p)
        cy = (cyf/HEIGHT)*ch_use
        if use_full_frame:
            fo = crop_full_frame(cb, s, cx, cy)
        else:
            fo = compose_frame(tb, crop_content_motion(cb, s, cx, cy))
        proc.stdin.write(fo.tobytes())
    proc.stdin.close(); proc.wait()

def render_classic(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                   style_mode="comic", language="vi"):
    # V10.3: English → không vẽ tay
    if language == "en" and style_mode == "comic":
        return render_kttv(image_path, duration, output_path, hand_path, motion, style_mode, language)

    tf = max(1, round(duration * FPS))
    df = int(max(1.5, min(duration-0.8, duration*DRAW_DURATION_RATIO))*FPS); rf = int(0.35*FPS)
    of = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    tb, cb = split_title_band(of)
    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)
    tr_full = extract_cont_traj(image_path); tr = traj_to_content(tr_full)
    ph = split_traj_phases(tr)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lt = tr[0] if tr else (WIDTH//2, ch_use//2)
    for fi in range(tf):
        hv = False
        if fi < df:
            if fi < pf[0]: cp=0; lp=fi/max(1,pf[0])
            elif fi < pf[1]: cp=1; lp=(fi-pf[0])/max(1,pf[1]-pf[0])
            else: cp=2; lp=(fi-pf[1])/max(1,pf[2]-pf[1])
            for pi in range(cp):
                for pt in ph[pi]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
            cps = ph[cp]
            if cps:
                cnt = max(1, min(int(lp*len(cps)), len(cps)))
                for pt in cps[:cnt]: cv2.circle(rm, pt, REVEAL_RADIUS, 255, -1)
                tg = cps[cnt-1]
            else: tg = lt
            hx = tg[0]+int(1.2*math.sin(fi*1.8)); hy = tg[1]+int(1.2*math.cos(fi*1.8))
            lt = (hx, hy); hv = True
        elif fi < df+rf:
            rm[:, :] = 255; pr = (fi-df)/max(1,rf)
            hx = int(lt[0]+(WIDTH+180-lt[0])*pr); hy = int(lt[1]+(ch_use+180-lt[1])*pr); hv = True
        else: rm[:, :] = 255
        a = (cv2.GaussianBlur(rm, (13, 13), 0).astype(np.float32)/255.0)[:, :, None]
        fc = (cb*a + wc*(1.0-a)).astype(np.uint8)
        if hv: paste_hand(fc, hb, ha, hx-tx, hy-ty)
        fo = compose_frame(tb, fc)
        proc.stdin.write(fo.tobytes())
    proc.stdin.close(); proc.wait()

def create_placeholder(out, title):
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    d = ImageDraw.Draw(img); f = font_for(48); text = "ẢNH KHÔNG TẠO ĐƯỢC"
    b = d.textbbox((0, 0), text, font=f); tw, th = b[2]-b[0], b[3]-b[1]
    d.text(((WIDTH-tw)/2, (HEIGHT-th)/2), text, fill="#cccccc", font=f)
    img.save(out, quality=95)

# ============================================================
# PARALLEL
# ============================================================
def parallel_gen(scenes, batch_dir, providers, image_timeout, progress_state,
                 flux_steps=4, fair_share=True, circuit=True, prio_fast=True,
                 enable_arrows=True, enable_shadow=True, style_mode="comic", language="vi"):
    if not providers: raise RuntimeError("Không có provider nào.")
    total_s = len(scenes)
    if fair_share:
        cap = max(3, int((total_s/len(providers))*FAIR_SHARE_MULTIPLIER))
        st.caption(f"⚖️ Fair share cap: **{cap}** cảnh/provider")
    else: cap = 999999
    if circuit: st.caption(f"🔌 Circuit breaker: {CIRCUIT_BREAKER_THRESHOLD} lần fail → loại")
    if prio_fast: st.caption(f"⚡ Slow penalty: >{SLOW_PROVIDER_THRESHOLD:.0f}s → sleep {SLOW_PROVIDER_PENALTY}s")

    q = Queue(); attempts = {}
    for i, s in enumerate(scenes):
        ir = batch_dir / f"scene_{i+1:03d}_raw.png"; im = batch_dir / f"scene_{i+1:03d}.jpg"
        if not im.exists():
            q.put((i, s, ir, im)); attempts[i] = 0
            progress_state["scene_status"][i] = {"status": "pending", "provider": None, "started": None, "elapsed": 0.0, "attempts": 0}
        else:
            progress_state["scene_status"][i] = {"status": "done", "provider": "(cached)", "started": None, "elapsed": 0.0, "attempts": 0}
    total = q.qsize()
    if total == 0: return {}, 0
    results = {}; failed = {}; rlock = threading.Lock()
    pstats = {p["name"]: {"ok": 0, "err": 0, "total_time": 0.0, "last_scene": None, "errors": [], "circuit_broken": False} for p in providers}

    def worker(pc):
        n = pc["name"]; my = 0; cf = 0; lok = 0; ltime = 0.0
        while my < cap:
            if circuit and cf >= CIRCUIT_BREAKER_THRESHOLD:
                with rlock: pstats[n]["circuit_broken"] = True
                return
            if prio_fast and lok >= 2 and (ltime/lok) > SLOW_PROVIDER_THRESHOLD:
                time.sleep(SLOW_PROVIDER_PENALTY)
            try: idx, s, ir, im = q.get_nowait()
            except Empty: return
            ca = attempts.get(idx, 0)
            if ca >= MAX_ATTEMPTS_PER_SCENE:
                create_placeholder(im, s["title"])
                add_comic_overlays(im, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""),
                                   s.get("callout_side", "right"), im, s.get("text_boxes", []),
                                   enable_arrows, enable_shadow, style_mode, language)
                with rlock:
                    results[idx] = "placeholder"; progress_state["done"] += 1
                    progress_state["scene_status"][idx] = {"status": "placeholder", "provider": "placeholder", "started": None, "elapsed": 0.0, "attempts": ca}
                continue
            t0 = time.time()
            with rlock:
                progress_state["scene_status"][idx] = {"status": "working", "provider": n, "started": t0, "elapsed": 0.0, "attempts": ca+1}
            try:
                kw = pc.get("kwargs", {}).copy()
                data = pc["fn"](s["visual_prompt"], *pc.get("args", []), timeout=image_timeout,
                                title=s.get("title", ""), callout_text=s.get("callout_text", ""),
                                language=language, **kw)
                if not data or len(data) < 500: raise RuntimeError("empty")
                save_image(data, ir)
                add_comic_overlays(ir, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""),
                                   s.get("callout_side", "right"), im, s.get("text_boxes", []),
                                   enable_arrows, enable_shadow, style_mode, language)
                el = time.time() - t0; my += 1; lok += 1; ltime += el; cf = 0
                with rlock:
                    results[idx] = n; pstats[n]["ok"] += 1; pstats[n]["total_time"] += el; pstats[n]["last_scene"] = idx+1
                    progress_state["scene_status"][idx] = {"status": "done", "provider": n, "started": t0, "elapsed": el, "attempts": ca+1}
                    progress_state["done"] += 1
            except Exception as e:
                el = time.time() - t0; err = str(e)[:150]; cf += 1
                with rlock:
                    attempts[idx] = ca+1; pstats[n]["err"] += 1; pstats[n]["errors"].append(err)
                    progress_state["scene_status"][idx] = {"status": "pending" if attempts[idx] < MAX_ATTEMPTS_PER_SCENE else "failed",
                        "provider": None if attempts[idx] < MAX_ATTEMPTS_PER_SCENE else n, "started": None, "elapsed": el, "attempts": attempts[idx], "error": err}
                if attempts[idx] < MAX_ATTEMPTS_PER_SCENE:
                    q.put((idx, s, ir, im)); time.sleep(FAIL_SLEEP_SECONDS)
                else:
                    create_placeholder(im, s["title"])
                    add_comic_overlays(im, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""),
                                       s.get("callout_side", "right"), im, s.get("text_boxes", []),
                                       enable_arrows, enable_shadow, style_mode, language)
                    with rlock:
                        results[idx] = "placeholder"; progress_state["done"] += 1
                        progress_state["scene_status"][idx] = {"status": "placeholder", "provider": "placeholder", "started": None, "elapsed": el, "attempts": attempts[idx]}

    with ThreadPoolExecutor(max_workers=len(providers)) as ex:
        fs = [ex.submit(worker, p) for p in providers]
        wait(fs, timeout=None)

    rem = []
    while not q.empty():
        try: rem.append(q.get_nowait())
        except Empty: break
    if rem:
        st.warning(f"⚠️ {len(rem)} cảnh sót, fallback tuần tự...")
        for idx, s, ir, im in rem:
            ok = False
            for pc in providers:
                try:
                    kw = pc.get("kwargs", {}).copy()
                    data = pc["fn"](s["visual_prompt"], *pc.get("args", []), timeout=min(image_timeout, 45),
                                    title=s.get("title", ""), callout_text=s.get("callout_text", ""),
                                    language=language, **kw)
                    if data and len(data) > 500:
                        save_image(data, ir)
                        add_comic_overlays(ir, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""),
                                           s.get("callout_side", "right"), im, s.get("text_boxes", []),
                                           enable_arrows, enable_shadow, style_mode, language)
                        with rlock:
                            results[idx] = pc["name"]; pstats[pc["name"]]["ok"] += 1; progress_state["done"] += 1
                            progress_state["scene_status"][idx] = {"status": "done", "provider": pc["name"]+" (fb)", "started": None, "elapsed": 0.0, "attempts": attempts.get(idx, 0)}
                        ok = True; break
                except Exception: continue
            if not ok:
                create_placeholder(im, s["title"])
                add_comic_overlays(im, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""),
                                   s.get("callout_side", "right"), im, s.get("text_boxes", []),
                                   enable_arrows, enable_shadow, style_mode, language)
                with rlock:
                    results[idx] = "placeholder"; progress_state["done"] += 1
                    progress_state["scene_status"][idx] = {"status": "placeholder", "provider": "placeholder", "started": None, "elapsed": 0.0, "attempts": attempts.get(idx, 0)}
    progress_state["provider_stats"] = pstats
    return results, len(failed)

# ============================================================
# RENDER BATCH
# ============================================================
def render_batch(batch_audio, scenes, batch_dir, hand_path, style,
                 cf_acc, cf_tok, hf_tok, fta_key, tg_key, nx_key, ag_key, pol_key, pol_mod,
                 image_timeout, flux_steps=4, fair_share=True, circuit=True, prio_fast=True,
                 chars=None, char_lock=None, enable_arrows=True, enable_shadow=True,
                 enable_sfx=True, sfx_vol=-12, enable_music=True, music_vol=-22,
                 seed_lock=None, style_mode="comic", language="vi"):
    total = len(scenes)
    if total == 0: raise RuntimeError("Không có cảnh nào.")
    chars = chars or {}; char_lock = char_lock or {}
    providers = build_provider_list(cf_acc, cf_tok, hf_tok, fta_key, tg_key, nx_key, ag_key,
                                    pol_key, pol_mod, flux_steps, chars, char_lock, seed_lock, style_mode)
    if not providers: raise RuntimeError("Chưa cấu hình provider.")

    native_text = (language == "en" and style_mode == "comic")
    mode_label = '👻 Horror' if style_mode=='horror' else ('🇬🇧 English full-frame' if native_text else '📚 Comic VI (title band)')
    st.markdown(f"### 🔗 {len(providers)} Provider ({mode_label})")
    pc = st.columns(min(4, len(providers)))
    for i, p in enumerate(providers):
        with pc[i % len(pc)]: st.markdown(f"**{i+1}.** {p['name']}")
    if char_lock:
        if "__global__" in char_lock:
            st.success(f"🌍 Global Lock: {char_lock['__global__'][:60]}...")
        others = [k for k in char_lock.keys() if k != "__global__"]
        if others: st.success(f"🔒 Per-name lock: {others}")
    if seed_lock is not None: st.info(f"🎲 Seed: {seed_lock}")

    st.markdown("### 🎨 Tạo ảnh song song")
    ps = {"done": 0, "scene_status": {}, "provider_stats": {p["name"]: {"ok":0,"err":0,"total_time":0.0,"last_scene":None,"errors":[],"circuit_broken":False} for p in providers}}
    plock = threading.Lock()
    bar = st.progress(0); txt = st.empty(); stats_t = st.empty(); scene_t = st.empty()

    def spd(avg):
        if avg <= 0: return "—"
        if avg < 5: return f"🚀 {avg:.1f}s"
        if avg < 15: return f"⚡ {avg:.1f}s"
        return f"🐢 {avg:.1f}s"

    def dash():
        with plock:
            done = ps["done"]; ss = dict(ps["scene_status"]); pst = dict(ps["provider_stats"])
        pct = min(1.0, done/total)*0.6
        bar.progress(pct); txt.markdown(f"**🎨 Ảnh: {done}/{total}** ({pct/0.6*100:.0f}%)")
        sd = []
        for p in providers:
            n = p["name"]; s = pst.get(n, {"ok":0,"err":0,"total_time":0.0,"last_scene":None,"errors":[],"circuit_broken":False})
            avg = s["total_time"]/s["ok"] if s["ok"]>0 else 0
            le = s.get("errors", [])[-1][:30] if s.get("errors") else "—"
            cb = "🔌 BROKEN" if s.get("circuit_broken") else "✅ OK"
            sd.append({"Provider": n, "✅ OK": s["ok"], "❌ Lỗi": s["err"], "Tốc độ": spd(avg),
                       "🎬 Cảnh cuối": f"#{s['last_scene']}" if s["last_scene"] else "—", "🔌 Circuit": cb, "🐛 Lỗi": le})
        if sd: stats_t.dataframe(sd, use_container_width=True, hide_index=True)
        rows = []
        for i in range(total):
            si = ss.get(i, {"status": "pending", "provider": None, "elapsed": 0.0, "attempts": 0})
            ic = {"pending": "⏳", "working": "🔄", "done": "✅", "failed": "❌", "placeholder": "⚠️"}.get(si["status"], "?")
            pv = si["provider"] or "—"
            el = f"{si.get('elapsed', 0):.1f}s" if si.get("elapsed", 0) > 0 else "—"
            at = si.get("attempts", 0)
            rows.append({"Cảnh": f"#{i+1:02d}", "Trạng thái": ic, "Provider": pv, "Thời gian": el, "Lần thử": f"{at}/{MAX_ATTEMPTS_PER_SCENE}"})
        scene_t.dataframe(rows, use_container_width=True, hide_index=True, height=min(400, 35*total+40))

    rc = {"r": None, "e": None}
    def run_p():
        try:
            r, _ = parallel_gen(scenes, batch_dir, providers, image_timeout, ps, flux_steps,
                                fair_share, circuit, prio_fast, enable_arrows, enable_shadow, style_mode, language)
            rc["r"] = r
        except Exception as e: rc["e"] = e
    t = threading.Thread(target=run_p, daemon=True); t.start()
    while t.is_alive():
        dash(); time.sleep(0.8)
    t.join(); dash()
    if rc["e"]: raise rc["e"]
    used = rc["r"] or {}
    st.success(f"✅ Đã tạo {len(used)}/{total} ảnh — {dict(Counter(used.values()))}")

    st.markdown("### 🎬 Render video")
    rb = st.progress(0); rt = st.empty()
    vids = []
    for i, s in enumerate(scenes, 1):
        im = batch_dir / f"scene_{i:03d}.jpg"; vd = batch_dir / f"scene_{i:03d}.mp4"
        if not im.exists(): raise RuntimeError(f"Thiếu ảnh scene {i}")
        dur = max(1.0, float(s["end"]) - float(s["start"]))
        mo = s.get("camera_motion", "zoom_in_center")
        if "1." in style or "Vẽ 3 phase" in style: render_kttv(im, dur, vd, hand_path, mo, style_mode, language)
        elif "2." in style or "Hybrid" in style: render_hybrid(im, dur, vd, hand_path, mo, style_mode, language)
        elif "3." in style or "Chỉ Camera" in style: render_pure(im, dur, vd, mo, style_mode, language)
        else: render_classic(im, dur, vd, hand_path, mo, style_mode, language)
        vids.append(vd)
        rb.progress(i/total); rt.markdown(f"**🎬 Render: {i}/{total}** — {s['title']}")

    cf_f = batch_dir / "concat.txt"
    cf_f.write_text("\n".join(f"file '{p.resolve()}'" for p in vids), encoding="utf-8")
    bv = batch_dir / "batch_video.mp4"
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(cf_f), "-c", "copy", "-movflags", "+faststart", str(bv)], timeout=900)

    fb = batch_dir / "batch_final.mp4"
    sfx_t = None; mus_t = None
    if enable_sfx:
        st.markdown("### 🔊 SFX")
        sfx_t = build_sfx_track(scenes, batch_dir / "sfx.wav", SFX_SAMPLE_RATE, sfx_vol)
    if enable_music:
        st.markdown("### 🎵 Nhạc nền")
        mus_t = build_music_track(scenes, batch_dir / "music.wav", SFX_SAMPLE_RATE, music_vol)

    if sfx_t or mus_t:
        mixed = batch_dir / "audio_mixed.m4a"
        mix_audio_tracks(str(batch_audio), str(sfx_t) if sfx_t else None, str(mus_t) if mus_t else None, str(mixed))
        run_cmd(["ffmpeg", "-y", "-i", str(bv), "-i", str(mixed), "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(fb)], timeout=900)
    else:
        run_cmd(["ffmpeg", "-y", "-i", str(bv), "-i", str(batch_audio), "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(fb)], timeout=900)

    for f in batch_dir.glob("scene_*_raw.png"): f.unlink(missing_ok=True)
    for f in batch_dir.glob("scene_*.mp4"): f.unlink(missing_ok=True)
    cf_f.unlink(missing_ok=True); bv.unlink(missing_ok=True)
    return fb

def concat_batches(vids, out):
    cf = out.parent / "batches.txt"
    cf.write_text("\n".join(f"file '{p.resolve()}'" for p in vids), encoding="utf-8")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(cf), "-c", "copy", "-movflags", "+faststart", str(out)], timeout=1800)

# ============================================================
# MAIN
# ============================================================
st.sidebar.divider()

if st.sidebar.button("🔎 KIỂM TRA PROVIDER", use_container_width=True):
    char_lock = build_character_lock(char_main_name, char_main_desc, char_second_name, char_second_desc, enable_char_lock)
    if enable_global_char and global_char_desc.strip():
        char_lock["__global__"] = global_char_desc.strip()
    providers = build_provider_list(cf_account, cf_token, hf_token, freetheai_key, together_key, nexa_key, agnes_key,
                                    pollinations_key, pollinations_model, flux_steps, {}, char_lock, None, style_mode)
    if not providers: st.error("Chưa có provider.")
    else:
        st.write(f"**{len(providers)} provider ({style_mode} mode):**")
        for i, p in enumerate(providers, 1): st.write(f"{i}. {p['name']}")
        if st.button("▶️ Test 1 ảnh"):
            if style_mode == "horror":
                tp = "2D dark horror illustration: a lone figure in a foggy hallway, moonlight, no text"
            elif effective_lang == "en":
                tp = "Colored cartoon illustration with warm earth tones, a caveman in a cave with campfire, brown and orange palette"
            else:
                tp = "2D comic doodle: a person at desk with laptop, white background, no text"
            for pc in providers:
                try:
                    with st.spinner(f"Test {pc['name']}..."):
                        t0 = time.time()
                        d = pc["fn"](tp, *pc.get("args", []), timeout=45,
                                    title="TEST TITLE", callout_text="HELLO WORLD",
                                    language=effective_lang, **pc.get("kwargs", {}))
                        el = time.time() - t0
                    if d and len(d) > 500:
                        img = Image.open(io.BytesIO(d)).convert("RGB").resize((WIDTH, HEIGHT))
                        st.success(f"✅ {pc['name']} — {el:.1f}s")
                        st.image(img, use_container_width=True); break
                except Exception as e:
                    st.warning(f"❌ {pc['name']}: {str(e)[:150]}")

audio = st.file_uploader("🎤 Tải lên voice", type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"])

if audio:
    st.audio(audio)
    if st.button("🚀 BẮT ĐẦU", type="primary", use_container_width=True):
        if not groq_key: st.error("Cần Groq API Key."); st.stop()

        lang_code = None
        if language_mode == "Tiếng Việt": lang_code = "vi"; effective_lang = "vi"
        elif language_mode == "English": lang_code = "en"; effective_lang = "en"
        else: lang_code = None
        st.info(f"🌐 Ngôn ngữ: **{language_mode}** | 🎨 Style: **{style_mode.upper()}** | ⏱️ Nhịp: **{scene_min}-{scene_max}s**")

        if use_script_mode == "Kết hợp voice + text": internal_mode = "combined"
        elif use_script_mode == "Chỉ dùng text": internal_mode = "text_only"
        else: internal_mode = "voice_only"

        char_lock = build_character_lock(char_main_name, char_main_desc, char_second_name, char_second_desc, enable_char_lock)
        if enable_global_char and global_char_desc.strip():
            char_lock["__global__"] = global_char_desc.strip()
            st.success(f"🌍 Global Lock: {global_char_desc[:60]}...")
        others = [k for k in char_lock.keys() if k != '__global__']
        if others: st.success(f"🔒 Per-name lock: {others}")
        seed_lock = random.randint(1, 2**31 - 1) if enable_seed_lock else None
        if seed_lock: st.info(f"🎲 Seed: {seed_lock}")

        cache_dir = Path.home() / ".wb_cache"; cache_dir.mkdir(exist_ok=True)

        root = Path(tempfile.mkdtemp(prefix=f"wb_v103_{style_mode}_"))
        try:
            src = root / audio.name; src.write_bytes(audio.getbuffer())
            dur = ffprobe_duration(src)
            effective_batch = 2 * 60 if style_mode == "horror" else BATCH_SECONDS
            st.info(f"Thời lượng: {dur/60:.2f} phút. Batch {effective_batch//60} phút.")

            client = groq_client(groq_key)
            bd = root / "batches"; bd.mkdir()
            chunks = chunk_audio(src, bd, effective_batch)
            hp = Path("hand.png")
            bvids = []; all_s = 0; stt = st.empty()

            vc = [(bi, ch, ffprobe_duration(ch)) for bi, ch in enumerate(chunks) if ffprobe_duration(ch) >= 5.0 or bi == 0]
            if not vc: st.error("Không có audio hợp lệ."); st.stop()

            cm_mode = ("random" if "Random" in camera_motion_mode else
                      f"fixed:{camera_motion_mode.replace('Cố định: ', '').strip()}"
                      if "Cố định" in camera_motion_mode else "auto")

            for idx, (bi, chunk, bdur) in enumerate(vc):
                bstart = bi * effective_batch
                stt.markdown(f"### 🧠 Đợt {idx+1}/{len(vc)} — STT...")
                tr = transcribe_file(client, chunk, stt_model, lang_code, cache_dir)
                if lang_code is None:
                    detected = detect_language(tr)
                    st.info(f"🌐 Whisper: **{detected}**")
                    effective_lang = "en" if detected.startswith("en") else "vi"
                segs = normalize_segments(tr, bstart)
                btext = "\n".join(f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}" for x in segs)

                tc = len(vc); bscript = ""
                if script_text and internal_mode in ("combined", "text_only"):
                    lines = [l.strip() for l in script_text.split("\n") if l.strip()]
                    lp = max(1, len(lines)//tc + 1)
                    s_l = idx*lp; e_l = min(s_l+lp, len(lines))
                    bscript = "\n".join(lines[s_l:e_l])

                stt.markdown(f"### ✂️ Đợt {idx+1}/{len(vc)} — Lên kịch bản ({effective_lang})...")
                scenes = make_scene_plan(client, btext, bstart, bdur, planner_model,
                                         scene_min, scene_max, max_scenes, cm_mode,
                                         language=effective_lang,
                                         enable_rich=enable_rich_overlay,
                                         char_lock=char_lock,
                                         enable_sfx=enable_sfx,
                                         enable_music=enable_music,
                                         user_script=bscript,
                                         use_script_mode=internal_mode,
                                         style_mode=style_mode)
                st.markdown(f"#### 📝 Đợt {idx+1}: {bdur:.1f}s → **{len(scenes)} cảnh**")
                with st.expander("Chi tiết", expanded=False):
                    for si, s in enumerate(scenes, 1):
                        ci = f" | [{s.get('callout_type','').upper()}]: \"{s.get('callout_text','')}\"" if s.get('callout_text') else ""
                        dur_s = s['end'] - s['start']
                        st.caption(f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s ({dur_s:.1f}s) — {s['title']}{ci} | 🎥 {s.get('camera_motion','?')} | 🔊 {s.get('sfx','none')} | 🎵 {s.get('music_emotion','none')}")

                bw = root / f"work_{idx+1:03d}"; bw.mkdir()
                stt.markdown(f"### 🎨 Đợt {idx+1}/{len(vc)} — Tạo ảnh + render...")
                bvi = render_batch(chunk, scenes, bw, hp, draw_style,
                                   cf_account, cf_token, hf_token, freetheai_key, together_key,
                                   nexa_key, agnes_key, pollinations_key, pollinations_model,
                                   image_timeout, flux_steps, fair_share_enabled, circuit_breaker_enabled,
                                   prioritize_fast, chars={}, char_lock=char_lock,
                                   enable_arrows=enable_arrows, enable_shadow=enable_shadow,
                                   enable_sfx=enable_sfx, sfx_vol=sfx_volume,
                                   enable_music=enable_music, music_vol=music_volume,
                                   seed_lock=seed_lock, style_mode=style_mode, language=effective_lang)
                sv = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bvi, sv); bvids.append(sv)
                all_s += len(scenes)
                shutil.rmtree(bw, ignore_errors=True)

            stt.markdown("### 🎬 Ghép video cuối...")
            fv = root / "video_final.mp4"
            concat_batches(bvids, fv)
            st.success(f"Hoàn thành! {all_s} cảnh ({style_mode} / {effective_lang} mode).")
            st.video(str(fv))
            st.download_button("⬇️ TẢI VIDEO", data=fv.read_bytes(),
                file_name="video_final.mp4", mime="video/mp4", use_container_width=True)
        except Exception as e:
            st.exception(e)
