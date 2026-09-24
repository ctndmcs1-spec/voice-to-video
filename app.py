"""
Xưởng Video Diễn Hoạt Kiến Thức AI — Bản Siêu Cấp V11.2
=======================================================
V11.2 FIX:
- Cache đúng voice, checkpoint ảnh/cảnh/batch, xem trước kịch bản
- Sửa timeline, không nhân đôi prompt, giữ phần audio đuôi
- FFmpeg writer an toàn, trộn âm thanh ưu tiên voice, xuất SRT
- Ngân sách token có thể chỉnh, chia batch theo sức chứa, xử lý 429 riêng
- English mode: KHÔNG tách title band, camera motion áp dụng full 720px
- English zoom nhẹ hơn (1.12) tránh cắt title
- VI/Horror giữ nguyên title band 95px + Pillow overlay
"""

import uuid
import textwrap
import os, re, io, json, math, time, base64, random, shutil, subprocess, tempfile, threading, wave, hashlib
from pathlib import Path
from queue import Queue, Empty
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
from studio_core import (VERSION, fingerprint, file_digest, read_json, write_json,
                         artifact_ok, mark_artifact, job_lock, RawVideoWriter,
                         frame_durations, script_slice, srt_text, planner_capacity, chat_completion, PlannerRateLimit, align_script_to_segments, scene_image_key, scene_narration)
import cv2
import numpy as np

# ============================================================
# CẤU HÌNH
# ============================================================
APP_TITLE = "Xưởng Video Diễn Hoạt Kiến Thức AI (V11.2)"
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
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title("🎬 Xưởng Video Diễn Hoạt Kiến Thức AI — V11.2")
st.caption("VOICE → KỊCH BẢN → HÌNH ẢNH → VIDEO • Studio V11.2")
st.markdown("Tạo video minh họa từ lời đọc, kiểm tra kịch bản trước khi tạo ảnh và tiếp tục khi bị gián đoạn.")

def setting(name):
    value = os.getenv(name)
    if value is not None: return value
    try: return str(st.secrets.get(name, ""))
    except Exception: return ""

with st.sidebar:
    st.header("🎨 Style Mode")
    style_mode_ui = st.radio("Phong cách video",
        ["📚 Kiến Thức (Comic)", "👻 Kinh Dị (Horror)"], index=0)
    style_mode = "horror" if "Horror" in style_mode_ui else "comic"

    st.header("🌐 Ngôn ngữ")
    language_mode = st.selectbox("Ngôn ngữ video",
        ["Auto Detect", "Tiếng Việt", "English"], index=0,
        help="English: AI vẽ chữ + full-frame camera. Vietnamese: Pillow overlay.")

    effective_lang = "vi"
    if language_mode == "English": effective_lang = "en"
    elif language_mode == "Tiếng Việt": effective_lang = "vi"

    if style_mode == "comic":
        if effective_lang == "en":
            st.success("🇬🇧 English: AI vẽ sticker title + full-frame camera (không title band)")
        else:
            st.info("🇻🇳 Vietnamese: Pillow overlay + title band 95px")
    else:
        st.warning("⚠️ Horror mode: nhịp 5-10s/cảnh. Khuyên dùng Pollinations flux-pro.")

    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key",
        value=setting("GROQ_API_KEY"), type="password")

    with st.expander("🎨 Nhà cung cấp ảnh AI", expanded=False):
        pollinations_key = st.text_input("Pollinations API Key",
            value=setting("POLLINATIONS_API_KEY"), type="password")
        pollinations_model = st.selectbox("Pollinations Model",
            ["flux-pro", "flux", "gptimage", "kontext", "flux-realism"], index=0)
        agnes_key = st.text_input("Agnes AI API Key",
            value=setting("AGNES_API_KEY"), type="password")
        cf_account = st.text_input("Cloudflare Account ID",
            value=setting("CLOUDFLARE_ACCOUNT_ID"), type="password")
        cf_token = st.text_input("Cloudflare API Token",
            value=setting("CLOUDFLARE_API_TOKEN"), type="password")
        hf_token = st.text_input("Hugging Face Token",
            value=setting("HF_TOKEN"), type="password")
        freetheai_key = st.text_input("FreeTheAi API Key",
            value=setting("FREETHEAI_API_KEY"), type="password")
        together_key = st.text_input("Together AI API Key",
            value=setting("TOGETHER_API_KEY"), type="password")
        nexa_key = st.text_input("NexaAPI Key",
            value=setting("NEXA_API_KEY"), type="password")

    st.header("📝 Văn bản kịch bản (tùy chọn)")
    script_text = st.text_area("Dán kịch bản", value="", height=100)
    use_script_mode = st.radio("Chế độ phân tích",
        ["Chỉ dùng voice", "Kết hợp voice + text", "Chỉ dùng text"], index=0)

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

    planner_output_budget = st.number_input("Token đầu ra tối đa / yêu cầu", min_value=256, max_value=14000,
        value=4096, step=64, help="Đây là ngân sách ứng dụng, không phải hạn mức tài khoản. Giảm khi Groq báo Request too large / OTPM. Tool tự chia đợt nhỏ hơn.")

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
    pacing = st.selectbox("Nhịp dựng", ["YouTube gọn (8–14s)", "Nhịp bản gốc"], index=0)
    if style_mode == "horror":
        default_min, default_max = 6, 10
    else:
        default_min, default_max = (8, 14) if pacing.startswith("YouTube") else (19, 27)
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

with st.sidebar:
    st.header("✨ Hoàn thiện video")
    consistent_provider = st.checkbox("Giữ một nhà cung cấp ảnh", value=True,
        help="Dùng provider đầu tiên trong danh sách để giảm thay đổi phong cách. Tắt để dùng lại chế độ nhiều provider song song.")
    allow_placeholder = st.checkbox("Cho phép ảnh dự phòng khi API lỗi", value=False)
    export_size = st.selectbox("Độ phân giải xuất", ["720p gốc", "1080p nâng kích thước"], index=0,
        help="Ảnh và diễn hoạt gốc 720p. 1080p được upscale, không tự bổ sung chi tiết thật.")
    normalize_voice = st.checkbox("Cân bằng âm lượng bản xuất", value=True)
    burn_subtitles = st.checkbox("Gắn phụ đề vào video", value=False)
    st.caption("Nhạc nền tự giảm khi có giọng đọc. SRT luôn có thể tải riêng.")

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

def groq_client(key): return Groq(api_key=key, max_retries=0, timeout=120.0)

def transcribe_file(client, path, model, language=None, cache_dir=None):
    # Content + STT settings identify a transcript; batch filenames are reused.
    audio_bytes = Path(path).read_bytes()
    settings = json.dumps({"version": 2, "model": model, "language": language}, sort_keys=True)
    key = hashlib.sha256(settings.encode() + b"\0" + audio_bytes).hexdigest()
    cf = Path(cache_dir) / f"transcript_v2_{key}.json" if cache_dir else None
    if cf and cf.exists():
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("segments"), list):
                st.info("💾 Đã dùng transcript của đúng nội dung audio và cấu hình STT.")
                return data
        except (ValueError, OSError):
            pass
    upload_bytes = audio_bytes
    upload_name = Path(path).name
    # Keep full-bandwidth batch audio for the video; shrink only the STT upload if needed.
    if len(upload_bytes) > 20 * 1024 * 1024:
        with tempfile.TemporaryDirectory() as temporary:
            speech = Path(temporary) / "speech.wav"
            run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1",
                     "-c:a", "pcm_s16le", str(speech)], timeout=300)
            upload_bytes = speech.read_bytes()
            upload_name = "speech.wav"
    kw = {"file": (upload_name, upload_bytes), "model": model,
          "response_format": "verbose_json", "timestamp_granularities": ["segment"],
          "temperature": 0.0}
    if language:
        kw["language"] = language
    result = client.audio.transcriptions.create(**kw)
    if cf:
        tmp = None
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            data = result.model_dump() if hasattr(result, "model_dump") else result
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cf.parent,
                                             delete=False, suffix=".tmp") as f:
                tmp = Path(f.name)
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, cf)
        except (OSError, TypeError, ValueError):
            pass
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
    return result

def detect_language(result):
    try:
        d = result.model_dump() if hasattr(result, "model_dump") else result
        return d.get("language", "unknown") if isinstance(d, dict) else getattr(result, "language", "unknown")
    except Exception: return "unknown"

def chunk_audio(src, out_dir, bs):
    pattern = str(Path(out_dir) / "batch_%03d.wav")
    run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:a:0", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
             "-f", "segment", "-segment_time", str(bs), "-reset_timestamps", "1", pattern], timeout=900)
    return sorted(Path(out_dir).glob("batch_*.wav"))

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

def smooth_crop(frame, scale, cx, cy):
    h, w = frame.shape[:2]
    scale = max(1.0, float(scale))
    crop_w, crop_h = w / scale, h / scale
    left = max(0.0, min(w-crop_w, cx-crop_w/2))
    top = max(0.0, min(h-crop_h, cy-crop_h/2))
    matrix = np.array([[1/scale, 0, left], [0, 1/scale, top]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

def crop_content_motion(c, scale, cx, cy):
    return smooth_crop(c, scale, cx, cy)

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
    try:
        _ensure_fonts()
        if noto.exists(): return ImageFont.truetype(str(noto), size)
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
        key = (emo, round(dur * sr))
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
    inputs = ["-i", str(voice)]
    filters = ["[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[voice][side]"]
    labels = ["[voice]"]; next_input = 1
    if music and Path(music).exists():
        inputs += ["-i", str(music)]
        filters.append(f"[{next_input}:a]aresample=48000,aformat=channel_layouts=stereo[music]")
        filters.append("[music][side]sidechaincompress=threshold=0.03:ratio=5:attack=15:release=300[ducked]")
        labels.append("[ducked]"); next_input += 1
    else:
        filters.append("[side]anullsink")
    if sfx and Path(sfx).exists():
        inputs += ["-i", str(sfx)]
        labels.append(f"[{next_input}:a]")
    filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0,alimiter=limit=0.95:level=0:latency=1[aout]")
    run_cmd(["ffmpeg", "-y", "-v", "error"] + inputs + ["-filter_complex", ";".join(filters),
             "-map", "[aout]", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(out)], timeout=600)
    return out

# ============================================================
# SCENE PLANNER
# ============================================================
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model,
                    min_s, max_s, max_scenes, camera_mode="auto",
                    language="vi", enable_rich=True, char_lock=None,
                    enable_sfx=True, enable_music=True,
                    user_script="", use_script_mode="voice_only", style_mode="comic", previous_scenes=None, feedback="", output_budget=4096):
    avg_dur = (min_s + max_s) / 2.0
    expected = min(max_scenes, max(1, round(batch_duration / avg_dur)))
    if use_script_mode != "text_only" and not transcript_text.strip():
        raise ValueError("Không nhận dạng được lời thoại. Hãy kiểm tra audio hoặc chọn ngôn ngữ cụ thể.")
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
TIMING: start/end tính bằng giây CỤC BỘ trong batch này, bắt đầu từ 0.
Các cảnh theo thứ tự thời gian, không chồng lấn, phủ kín audio đến {batch_duration:.2f}s.
Mỗi cảnh bám đúng lời thoại trong khoảng thời gian tương ứng.
Không lặp visual_prompt và không kể lại nội dung trước đó.
DỰNG PHIM: mỗi cảnh truyền tải một ý, chọn cỡ cảnh toàn/trung/cận phù hợp.
Luân phiên bối cảnh, hành động và cỡ cảnh; giữ ngoại hình nhân vật, bảng màu nhất quán.
Mở đoạn đầu bằng hình ảnh cụ thể gắn với câu hook, tránh hình minh họa chung chung.
Không tự bịa số liệu, thương hiệu, biểu đồ hoặc thông tin không có trong lời thoại.
Giữ phần chữ ngắn, tránh vùng mép ảnh và vùng dưới cùng dành cho phụ đề.
Các cảnh trước (chỉ tham khảo tính liên tục, KHÔNG kể lại): {json.dumps(previous_scenes or [], ensure_ascii=False)}
Lỗi cần sửa từ lần lập kế hoạch trước: {feedback}
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
        user = (f"Text đã đối chiếu theo lời đọc. Dùng mốc WHISPER TIMING cho từng ý, không phân bố đều text theo thời lượng.\nAudio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\n"
                f"USER SCRIPT:\n{user_script}\n\nWHISPER TIMING:\n{transcript_text}")
    elif use_script_mode == "text_only" and user_script.strip():
        user = f"Audio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\nSCRIPT:\n{user_script}"
    else:
        user = f"Audio length: {batch_duration:.2f}s.\nMAX {expected} SCENES.\n\nTRANSCRIPT:\n{transcript_text}"

    if output_budget < 1536:
        system += "\nNgân sách thấp: JSON gọn không markdown/giải thích. visual_prompt tối đa 40 từ, tiêu đề tối đa 4 từ, callout tối đa 6 từ, text_boxes tối đa 2 mục ngắn."
    dyn_max = min(int(output_budget), max(256, 512 + expected * 650))
    raw = ""
    try:
        response = chat_completion(client, model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            dyn_max, notify=st.info)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError("JSON bị cắt vì hết token. Giảm số cảnh/đợt hoặc điều chỉnh ngân sách token phù hợp Limits.")
        raw = choice.message.content or ""
        obj = extract_json(raw); raw_scenes = obj.get("scenes", [])
    except PlannerRateLimit:
        raise
    except Exception as exc:
        # Authentication/model availability/network errors are not script repair requests.
        if getattr(exc, "status_code", None) is not None: raise
        raise RuntimeError(f"Không tạo được kịch bản hợp lệ: {str(exc)[:300]}") from exc
    if not isinstance(raw_scenes, list) or len(raw_scenes) > max_scenes:
        raise ValueError("scenes phải là danh sách trong giới hạn số cảnh đã chọn.")

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
    for s in raw_scenes:
        try:
            a = float(s["start"]); b = float(s["end"])
            if not math.isfinite(a) or not math.isfinite(b): continue
            a = max(0.0, a); b = min(batch_duration, b)
            if b <= a: continue
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
                "visual_prompt": vp, "text_boxes": clean_tbs(s.get("text_boxes", [])) if enable_rich else []})
        except Exception: continue

    if len(clean) != len(raw_scenes):
        raise ValueError("Có cảnh sai thời gian hoặc thiếu mô tả; cần lập lại kịch bản.")
    if not clean:
        raise ValueError("Kịch bản không có cảnh hợp lệ. Dừng thay vì dùng hai prompt lặp lại.")

    # Sort and reject overlapping intervals before closing timing gaps.
    clean.sort(key=lambda scene: (scene["start"], scene["end"]))
    ordered = []
    prompts = set()
    for scene in clean:
        signature = " ".join(scene["visual_prompt"].casefold().split())
        if signature in prompts:
            raise ValueError("AI trả về mô tả ảnh trùng nhau. Hãy thử lên kịch bản lại.")
        prompts.add(signature)
        if ordered and (scene["start"] <= ordered[-1]["start"] or scene["start"] < ordered[-1]["end"] - 0.05):
            raise ValueError("Kịch bản có cảnh chồng lấn; dừng để tránh sai thứ tự lời thoại.")
        ordered.append(scene)
    ordered[0]["start"] = 0.0
    for i in range(len(ordered) - 1):
        ordered[i]["end"] = ordered[i + 1]["start"]
    ordered[-1]["end"] = batch_duration
    # Never split a long scene by copying its visual_prompt and regenerating the image.
    final = ordered
    if any(scene["end"] - scene["start"] > max_s * 1.3 for scene in final):
        st.warning("Một số cảnh dài hơn nhịp đã chọn. Giữ một cảnh, không nhân đôi ảnh.")

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
    title_color = "#fff1e6" if is_horror else "#111111"
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
        callout_text = "\n".join(textwrap.wrap(callout_text, width=26, break_long_words=False)[:3])
        f_text = font_for(28, bold=True)
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

# English comic: zoom nhẹ để không cắt title
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
    if style_mode == "horror":
        src = HORROR_MOTIONS
    elif language == "en":
        src = EN_COMIC_MOTIONS
    else:
        src = COMIC_MOTIONS
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

def crop_full_frame(frame_bgr, scale, cx, cy):
    """Crop full frame (không tách title band) — dùng cho English mode."""
    return smooth_crop(frame_bgr, scale, cx, cy)

# ============================================================
# RENDER 4 STYLES — V11.2: English full-frame, VI/Horror title band
# ============================================================
def render_kttv(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                style_mode="comic", language="vi"):
    tf = max(1, round(duration * FPS))
    dd = min(max(0.0, duration - 1.0 / FPS), max(0.0, duration * DRAW_DURATION_RATIO))
    df = int(dd * FPS); rf = min(int(0.35 * FPS), max(0, tf-df-1))
    of = cv2.imread(str(image_path))
    if of is None: raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    of = cv2.resize(of, (WIDTH, HEIGHT))

    use_full_frame = (language == "en" and style_mode == "comic")
    if use_full_frame:
        tb = None; cb = of
    else:
        tb, cb = split_title_band(of)

    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)

    zt = extract_stag_traj(image_path)
    ap_full = [p for z in zt for p in z] or [(WIDTH//2, HEIGHT//2)]
    if use_full_frame:
        ap = [(px, py) for (px, py) in ap_full]
    else:
        ap = traj_to_content(ap_full)
    ph = split_traj_phases(ap)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path)]
    with RawVideoWriter(cmd) as proc:
        lt = ap[0] if ap else (WIDTH//2, ch_use//2)
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
            if fi < df+rf:
                sc, cu, cyu = 1.0, WIDTH*0.5, ch_use*0.5
            else:
                op = (fi-df-rf)/max(1, tf-df-rf)
                st_, cx_, cy_ = interp_motion(kfs, op)
                # cy_ được tính theo HEIGHT gốc → scale theo ch_use
                cyc = (cy_/HEIGHT)*ch_use
                bl = ease(min(1.0, op*1.8))
                sc = 1.0 + (st_-1.0)*bl
                cu = WIDTH*0.5 + (cx_-WIDTH*0.5)*bl
                cyu = ch_use*0.5 + (cyc-ch_use*0.5)*bl
            if use_full_frame:
                fo = crop_full_frame(fc, sc, cu, cyu)
            else:
                fo = compose_frame(tb, crop_content_motion(fc, sc, cu, cyu))
            proc.stdin.write(fo.tobytes())

def render_hybrid(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                  style_mode="comic", language="vi"):
    tf = max(1, round(duration * FPS))
    dd = min(max(0.0, duration - 1.0 / FPS), max(0.0, duration * DRAW_DURATION_RATIO))
    df = int(dd*FPS); rf = min(int(0.35*FPS), max(0, tf-df-1))
    of = cv2.imread(str(image_path))
    if of is None: raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    of = cv2.resize(of, (WIDTH, HEIGHT))

    use_full_frame = (language == "en" and style_mode == "comic")
    if use_full_frame:
        tb = None; cb = of
    else:
        tb, cb = split_title_band(of)

    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)
    tr_full = extract_cont_traj(image_path)
    tr = [(px, py) for (px, py) in tr_full] if use_full_frame else traj_to_content(tr_full)
    ph = split_traj_phases(tr)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path)]
    with RawVideoWriter(cmd) as proc:
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
            if use_full_frame:
                fo = crop_full_frame(fc, sc, cu, cyu)
            else:
                cw = int(WIDTH/sc); chh = int(ch_use/sc)
                ccx = max(cw//2, min(WIDTH-cw//2, int(cu))); ccy = max(chh//2, min(ch_use-chh//2, int(cyu)))
                fo = compose_frame(tb, crop_content_motion(fc, sc, ccx, ccy))
            proc.stdin.write(fo.tobytes())

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
           "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path)]
    with RawVideoWriter(cmd) as proc:
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

def render_classic(image_path, duration, output_path, hand_path, motion="zoom_in_center",
                   style_mode="comic", language="vi"):
    tf = max(1, round(duration * FPS))
    df = min(tf - 1, int(duration * DRAW_DURATION_RATIO * FPS)); rf = min(int(0.35*FPS), max(0, tf-df-1))
    of = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    use_full_frame = (language == "en" and style_mode == "comic")
    if use_full_frame:
        tb = None; cb = of
    else:
        tb, cb = split_title_band(of)
    ch_use = cb.shape[0]
    wc = np.full_like(cb, 255); rm = np.zeros((ch_use, WIDTH), dtype=np.uint8)
    tr_full = extract_cont_traj(image_path)
    tr = [(px, py) for (px, py) in tr_full] if use_full_frame else traj_to_content(tr_full)
    ph = split_traj_phases(tr)
    pf = [int(df*PHASE_RATIOS[0]), int(df*(PHASE_RATIOS[0]+PHASE_RATIOS[1])), df]
    hb, ha, tx, ty = load_hand(hand_path)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}",
           "-pix_fmt", "bgr24", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path)]
    with RawVideoWriter(cmd) as proc:
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
            if use_full_frame:
                fo = fc
            else:
                fo = compose_frame(tb, fc)
            proc.stdin.write(fo.tobytes())

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
    lock = progress_state.setdefault("_lock", threading.Lock())
    stats = progress_state["provider_stats"]
    results = {}; attempts = {}; failures = {}; q = Queue()
    cap = max(3, math.ceil(len(scenes)/len(providers)*FAIR_SHARE_MULTIPLIER)) if fair_share else len(scenes)
    keys = {}
    for i, scene in enumerate(scenes):
        image = batch_dir / f"scene_{i+1:03d}.jpg"
        keys[i] = scene_image_key(scene, enable_arrows, enable_shadow, style_mode, language)
        if artifact_ok(image, keys[i]):
            results[i] = "cached"
            progress_state["done"] += 1
            progress_state["scene_status"][i] = {"status": "done", "provider": "cached", "attempts": 0}
        else:
            q.put(i); attempts[i] = 0
            progress_state["scene_status"][i] = {"status": "pending", "provider": None, "attempts": 0}

    def worker(provider):
        name = provider["name"]; consecutive = 0; completed = 0
        while completed < cap:
            if circuit and consecutive >= CIRCUIT_BREAKER_THRESHOLD:
                with lock: stats[name]["circuit_broken"] = True
                return
            try: i = q.get_nowait()
            except Empty: return
            if attempts[i] >= MAX_ATTEMPTS_PER_SCENE:
                failures[i] = "Đã hết số lần thử"; continue
            scene = scenes[i]; started = time.monotonic()
            with lock:
                attempts[i] += 1
                progress_state["scene_status"][i] = {"status": "working", "provider": name, "attempts": attempts[i]}
            image = batch_dir / f"scene_{i+1:03d}.jpg"
            raw = batch_dir / f"scene_{i+1:03d}_raw.png"
            try:
                data = provider["fn"](scene["visual_prompt"], *provider.get("args", []),
                      timeout=image_timeout, title=scene.get("title", ""),
                      callout_text=scene.get("callout_text", ""), language=language,
                      **provider.get("kwargs", {}))
                save_image(data, raw)
                add_comic_overlays(raw, scene["title"], scene.get("callout_type", "speech"),
                      scene.get("callout_text", ""), scene.get("callout_side", "right"), image,
                      scene.get("text_boxes", []), enable_arrows, enable_shadow, style_mode, language)
                mark_artifact(image, keys[i])
                elapsed = time.monotonic()-started
                with lock:
                    results[i] = name; progress_state["done"] += 1
                    stats[name]["ok"] += 1; stats[name]["total_time"] += elapsed; stats[name]["last_scene"] = i+1
                    progress_state["scene_status"][i] = {"status": "done", "provider": name, "attempts": attempts[i], "elapsed": elapsed}
                completed += 1; consecutive = 0
                if prio_fast and elapsed > SLOW_PROVIDER_THRESHOLD: time.sleep(SLOW_PROVIDER_PENALTY)
            except Exception as exc:
                consecutive += 1
                with lock:
                    stats[name]["err"] += 1; stats[name]["errors"].append(str(exc)[:150])
                    progress_state["scene_status"][i] = {"status": "failed", "provider": name, "attempts": attempts[i]}
                if attempts[i] < MAX_ATTEMPTS_PER_SCENE:
                    q.put(i); time.sleep(FAIL_SLEEP_SECONDS)
                else: failures[i] = str(exc)

    # Fair-share rounds never bypass the circuit breaker or the per-scene retry cap.
    while not q.empty():
        active = [p for p in providers if not stats[p["name"]]["circuit_broken"]]
        if not active: break
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [executor.submit(worker, p) for p in active]
            for future in futures: future.result()
    for i, scene in enumerate(scenes):
        if i in results: continue
        image = batch_dir / f"scene_{i+1:03d}.jpg"
        create_placeholder(image, scene["title"])
        # No success marker: failed images will be retried when resuming.
        Path(str(image)+".json").unlink(missing_ok=True)
        results[i] = "placeholder"; failures.setdefault(i, "Provider không khả dụng")
        with lock:
            progress_state["done"] += 1
            progress_state["scene_status"][i] = {"status": "placeholder", "provider": "placeholder", "attempts": attempts.get(i, 0)}
    return results, len(failures)

# ============================================================
# RENDER BATCH
# ============================================================
def render_batch(batch_audio, scenes, batch_dir, hand_path, style,
                 cf_acc, cf_tok, hf_tok, fta_key, tg_key, nx_key, ag_key, pol_key, pol_mod,
                 image_timeout, flux_steps=4, fair_share=True, circuit=True, prio_fast=True,
                 chars=None, char_lock=None, enable_arrows=True, enable_shadow=True,
                 enable_sfx=True, sfx_vol=-12, enable_music=True, music_vol=-22,
                 seed_lock=None, style_mode="comic", language="vi", allow_placeholder=False, consistent_provider=True):
    total = len(scenes)
    if total == 0: raise RuntimeError("Không có cảnh nào.")
    chars = chars or {}; char_lock = char_lock or {}
    providers = build_provider_list(cf_acc, cf_tok, hf_tok, fta_key, tg_key, nx_key, ag_key,
                                    pol_key, pol_mod, flux_steps, chars, char_lock, seed_lock, style_mode)
    if not providers: raise RuntimeError("Chưa cấu hình provider.")
    if consistent_provider: providers = providers[:1]

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
    plock = threading.Lock(); ps["_lock"] = plock
    bar = st.progress(0); txt = st.empty(); stats_t = st.empty(); scene_t = st.empty()
    image_preview = st.empty(); previewed = set()

    def spd(avg):
        if avg <= 0: return "—"
        if avg < 5: return f"🚀 {avg:.1f}s"
        if avg < 15: return f"⚡ {avg:.1f}s"
        return f"🐢 {avg:.1f}s"

    def dash():
        with plock:
            done = ps["done"]; ss = dict(ps["scene_status"]); pst = json.loads(json.dumps(ps["provider_stats"]))
        fresh = [index for index, status in ss.items() if status["status"] == "done" and index not in previewed]
        if fresh:
            index = fresh[-1]
            picture = batch_dir / f"scene_{index+1:03d}.jpg"
            if picture.exists():
                image_preview.image(str(picture), caption=f"Cảnh {index+1}/{total}: {scenes[index]['title']}", use_container_width=True)
            previewed.update(fresh)
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
    try:
        while t.is_alive():
            dash(); time.sleep(0.8)
    finally:
        t.join()  # Keep the job lock until image workers have finished on rerun.
    dash()
    if rc["e"]: raise rc["e"]
    used = rc["r"] or {}
    missing = [i+1 for i, provider in used.items() if provider == "placeholder"]
    if missing and not allow_placeholder:
        raise RuntimeError(f"Chưa có ảnh thật cho cảnh {missing}. Tiến độ đã lưu; bấm BẮT ĐẦU để thử lại các cảnh lỗi.")
    st.success(f"✅ Ảnh: {len(used)}/{total} — {dict(Counter(used.values()))}")
    batch_key = fingerprint({"version": VERSION, "scenes": scenes, "images": [file_digest(batch_dir / f"scene_{i+1:03d}.jpg") for i in range(total)],
                            "style": style, "language": language, "audio": file_digest(batch_audio),
                            "hand": file_digest(hand_path) if hand_path.exists() else "fallback", "sfx": [enable_sfx, sfx_vol], "music": [enable_music, music_vol]})
    if artifact_ok(batch_dir / "batch_final.mp4", batch_key):
        st.info("💾 Đợt này đã hoàn thành, dùng lại video đã lưu.")
        return batch_dir / "batch_final.mp4"

    st.markdown("### 🎬 Render video")
    rb = st.progress(0); rt = st.empty()
    vids = []; durations = frame_durations(scenes, FPS)
    for i, s in enumerate(scenes, 1):
        im = batch_dir / f"scene_{i:03d}.jpg"; vd = batch_dir / f"scene_{i:03d}.mp4"
        if not im.exists(): raise RuntimeError(f"Thiếu ảnh scene {i}")
        dur = durations[i-1]
        mo = s.get("camera_motion", "zoom_in_center")
        video_key = fingerprint({"version": VERSION, "image": file_digest(im), "duration": dur,
                                 "style": style, "motion": mo, "mode": style_mode, "language": language,
                                 "hand": file_digest(hand_path) if hand_path.exists() else "fallback"})
        if not artifact_ok(vd, video_key):
            if "1." in style or "Vẽ 3 phase" in style: render_kttv(im, dur, vd, hand_path, mo, style_mode, language)
            elif "2." in style or "Hybrid" in style: render_hybrid(im, dur, vd, hand_path, mo, style_mode, language)
            elif "3." in style or "Chỉ Camera" in style: render_pure(im, dur, vd, mo, style_mode, language)
            else: render_classic(im, dur, vd, hand_path, mo, style_mode, language)
            mark_artifact(vd, video_key)
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

    mixed = batch_dir / "audio_mixed.m4a"
    mix_audio_tracks(str(batch_audio), str(sfx_t) if sfx_t else None, str(mus_t) if mus_t else None, str(mixed))
    run_cmd(["ffmpeg", "-y", "-v", "error", "-i", str(bv), "-i", str(mixed), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "copy", "-t", str(scenes[-1]["end"]), "-movflags", "+faststart", str(fb)], timeout=900)

    mark_artifact(fb, batch_key)
    cf_f.unlink(missing_ok=True); bv.unlink(missing_ok=True)
    return fb

def concat_batches(vids, out):
    cf = out.parent / "batches.txt"
    cf.write_text("\n".join(f"file '{p.resolve()}'" for p in vids), encoding="utf-8")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(cf), "-c", "copy", "-movflags", "+faststart", str(out)], timeout=1800)

# ============================================================
# PROJECT WORKFLOW — session-isolated checkpoints, no stored API keys
# ============================================================
def validate_storyboard(batches):
    seen = set()
    for batch in batches:
        for scene in batch["scenes"]:
            prompt = str(scene.get("visual_prompt", "")).strip()
            if len(prompt) < 10: raise ValueError("Mô tả ảnh phải có ít nhất 10 ký tự.")
            if not str(scene.get("title", "")).strip(): raise ValueError("Tiêu đề cảnh không được để trống.")
            signature = " ".join(prompt.casefold().split())
            if signature in seen: raise ValueError("Có mô tả ảnh trùng trong dự án. Hãy đổi bố cục/hành động.")
            seen.add(signature)


def loudness_filter(path):
    # Measure first: silent/very short audio can yield infinite loudnorm gains.
    cmd = ["ffmpeg", "-v", "info", "-i", str(path), "-vn", "-af",
           "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
    if result.returncode: raise RuntimeError(result.stderr[-2000:])
    matches = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', result.stderr)
    if not matches: raise RuntimeError("Không đo được âm lượng audio.")
    stats = json.loads(matches[-1])
    keys = ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]
    if not all(math.isfinite(float(stats[key])) for key in keys):
        return "anull"
    return (f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={stats['input_i']}:"
            f"measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
            f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true")


def finish_export(batches, root, duration, captions):
    joined = root / "joined.mp4"
    concat_batches(batches, joined)
    final = root / "video_final.mp4"; part = root / "export.part.mp4"
    filters = []
    if "1080p" in export_size: filters.append("scale=1920:1080:flags=lanczos")
    # Repeat only the last frame to cover sub-frame/AAC rounding, never an entire scene.
    filters.append("tpad=stop_mode=clone:stop_duration=1")
    if burn_subtitles and captions.strip():
        subtitles = (root / "subtitles.srt").as_posix()
        filters.append(f"subtitles=filename='{subtitles}':force_style='FontName=DejaVu Sans,FontSize=20,Outline=2,MarginV=24'")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(joined), "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-af", (loudness_filter(joined)+"," if normalize_voice else "")+"apad",
           "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
           "-t", str(duration), "-movflags", "+faststart", str(part)]
    run_cmd(cmd, timeout=max(1800, int(duration*20)))
    os.replace(part, final)
    return final


st.sidebar.divider()
if st.sidebar.button("🔎 KIỂM TRA PROVIDER", use_container_width=True):
    st.session_state["show_provider_test"] = not st.session_state.get("show_provider_test", False)
if st.session_state.get("show_provider_test"):
    test_lock = build_character_lock(char_main_name, char_main_desc, char_second_name, char_second_desc, enable_char_lock)
    if enable_global_char and global_char_desc.strip(): test_lock["__global__"] = global_char_desc.strip()
    test_providers = build_provider_list(cf_account, cf_token, hf_token, freetheai_key, together_key, nexa_key,
        agnes_key, pollinations_key, pollinations_model, flux_steps, {}, test_lock, None, style_mode)
    st.write("Provider đã cấu hình: " + (", ".join(p["name"] for p in test_providers) or "Chưa có"))
    if st.button("▶️ Test 1 ảnh", disabled=not test_providers):
        for provider in test_providers:
            try:
                with st.spinner(f"Test {provider['name']}..."):
                    data = provider["fn"]("A cinematic illustration of a character beside a desk, coherent colors",
                        *provider.get("args", []), timeout=image_timeout, language=effective_lang,
                        title="TEST", callout_text="", **provider.get("kwargs", {}))
                    st.image(Image.open(io.BytesIO(data)), use_container_width=True)
                    st.success(f"Provider hoạt động: {provider['name']}"); break
            except Exception as exc: st.warning(f"{provider['name']}: {str(exc)[:150]}")

audio = st.file_uploader("🎤 Tải lên voice", type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"])
if audio:
    st.audio(audio)
    internal_mode = {"Chỉ dùng voice": "voice_only", "Kết hợp voice + text": "combined", "Chỉ dùng text": "text_only"}[use_script_mode]
    char_lock = build_character_lock(char_main_name, char_main_desc, char_second_name, char_second_desc, enable_char_lock)
    if enable_global_char and global_char_desc.strip(): char_lock["__global__"] = global_char_desc.strip()
    lang_code = {"Tiếng Việt": "vi", "English": "en"}.get(language_mode)
    cm_mode = ("random" if "Random" in camera_motion_mode else
               f"fixed:{camera_motion_mode.replace('Cố định: ', '').strip()}" if "Cố định" in camera_motion_mode else "auto")
    # Credentials never enter the persisted project data or fingerprint.
    config = {"version": VERSION, "audio": hashlib.sha256(audio.getbuffer()).hexdigest(),
        "suffix": Path(audio.name).suffix.lower(), "mode": internal_mode,
        "script": script_text if internal_mode != "voice_only" else "",
        "language": language_mode, "style_mode": style_mode, "char_lock": char_lock,
        "seed_lock": enable_seed_lock, "stt_model": stt_model, "planner_model": planner_model, "output_budget": int(planner_output_budget),
        "scene_min": scene_min, "scene_max": scene_max, "max_scenes": max_scenes,
        "camera": cm_mode, "rich": enable_rich_overlay, "arrows": enable_arrows, "shadow": enable_shadow,
        "sfx": [enable_sfx, sfx_volume], "music": [enable_music, music_volume], "render": draw_style,
        "provider_flags": [bool(cf_account and cf_token), bool(agnes_key), bool(together_key), bool(freetheai_key),
                           bool(hf_token), bool(nexa_key), bool(pollinations_key)],
        "image_model": pollinations_model, "steps": flux_steps, "consistent": consistent_provider}
    if "studio_session" not in st.session_state: st.session_state["studio_session"] = uuid.uuid4().hex
    project_id = fingerprint(config)
    root = Path(tempfile.gettempdir()) / "voice_video_v11" / st.session_state["studio_session"] / project_id
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "storyboard.json"
    project = read_json(plan_path)
    actions = st.columns(2)
    plan_only = actions[0].button("📝 1. Xem kịch bản trước", use_container_width=True)
    start = actions[1].button("🚀 BẮT ĐẦU / TIẾP TỤC", type="primary", use_container_width=True)
    st.caption("Giữ nguyên voice và thiết lập để tiếp tục các cảnh chưa xong. Không cần tạo lại cảnh đã lưu trong phiên này.")

    if plan_only or start:
        if not groq_key: st.error("Cần Groq API Key."); st.stop()
        if internal_mode in ("combined", "text_only") and not script_text.strip():
            st.error("Hãy nhập kịch bản cho chế độ đã chọn."); st.stop()
        try:
            with job_lock(root):
                source = root / ("source" + Path(audio.name).suffix.lower())
                if not source.exists(): source.write_bytes(audio.getbuffer())
                duration = ffprobe_duration(source)
                if duration <= 0: raise ValueError("Audio không có thời lượng hợp lệ.")
                if project is None:
                    project = {"config": config, "duration": duration, "seed": random.randint(1, 2**31-1) if enable_seed_lock else None,
                               "batches": [], "complete": False}
                effective_batch = min(120 if style_mode == "horror" else BATCH_SECONDS,
                    max(5, int(min(max_scenes, planner_capacity(planner_output_budget)) * (scene_min+scene_max)/2)))
                chunks_dir = root / "batches"; chunks_dir.mkdir(exist_ok=True)
                chunk_key = fingerprint([config["audio"], effective_batch, VERSION])
                chunk_manifest = read_json(chunks_dir / "complete.json", {})
                chunks = sorted(chunks_dir.glob("batch_*.wav"))
                if chunk_manifest.get("key") != chunk_key or not chunks or len(chunks) != chunk_manifest.get("count"):
                    for old in chunks: old.unlink()
                    chunks = chunk_audio(source, chunks_dir, effective_batch)
                    write_json(chunks_dir / "complete.json", {"key": chunk_key, "count": len(chunks)})
                client = groq_client(groq_key); offset = 0.0
                aligned_batches = None
                if internal_mode == "combined":
                    # Align the entire script once, so pauses/speaking speed cannot shift batch text.
                    alignment_path = root / "alignment.json"
                    aligned_batches = read_json(alignment_path)
                    if aligned_batches is None:
                        heard = []; records = []; absolute = 0.0
                        for chunk in chunks:
                            length = ffprobe_duration(chunk)
                            st.info(f"Đối chiếu text + voice: nhận dạng {len(records)+1}/{len(chunks)}")
                            transcript = transcribe_file(client, chunk, stt_model, lang_code, root / "transcripts")
                            language = lang_code or ("en" if str(detect_language(transcript)).lower().startswith("en") else "vi")
                            local = normalize_segments(transcript, 0.0)
                            begin = len(heard)
                            for segment in local:
                                segment = dict(segment, start=max(0.0, segment["start"]), end=min(length, segment["end"]))
                                if segment["end"] > segment["start"]:
                                    heard.append(dict(segment, start=absolute+segment["start"], end=absolute+segment["end"]))
                            records.append({"offset":absolute,"duration":length,"language":language,"begin":begin,"end":len(heard)})
                            absolute += length
                        corrected, diagnostics = align_script_to_segments(script_text, heard)
                        aligned_batches = {"diagnostics":diagnostics,"batches":[]}
                        for record in records:
                            local = [dict(segment, start=segment["start"]-record["offset"], end=segment["end"]-record["offset"])
                                     for segment in corrected[record["begin"]:record["end"]]]
                            aligned_batches["batches"].append({"language":record["language"],"segments":local})
                        write_json(alignment_path, aligned_batches)
                    diagnostic = aligned_batches["diagnostics"]
                    st.info(f"Text/voice khớp từ {diagnostic['score']:.0%}; {diagnostic['review_segments']} đoạn cần xem lại. Mốc thời gian giữ theo Whisper.")
                    if diagnostic["script_only_words"]:
                        st.warning("Các từ có trong text nhưng chưa tìm được thời gian trong voice: " + diagnostic["script_only_words"][:500])
                st.info(f"Voice: {duration/60:.2f} phút • {len(chunks)} đợt • Dự án {project_id[:8]}")
                for index, chunk in enumerate(chunks):
                    batch_duration = ffprobe_duration(chunk)
                    if index < len(project["batches"]):
                        offset += batch_duration; continue
                    st.markdown(f"### 📝 Lập kịch bản đợt {index+1}/{len(chunks)}")
                    batch_script = script_slice(script_text, offset, offset+batch_duration, duration) if internal_mode == "text_only" else ""
                    if internal_mode == "combined":
                        aligned = aligned_batches["batches"][index]
                        language = aligned["language"]; segments = aligned["segments"]
                        batch_script = "\n".join(segment["text"] for segment in segments)
                    elif internal_mode == "text_only":
                        language = lang_code or ("vi" if re.search(r"[À-ỹ]", script_text) else "en")
                        segments = [{"start": 0.0, "end": batch_duration, "text": batch_script}]
                    else:
                        transcript = transcribe_file(client, chunk, stt_model, lang_code, root / "transcripts")
                        language = lang_code or ("en" if str(detect_language(transcript)).lower().startswith("en") else "vi")
                        segments = normalize_segments(transcript, 0.0)
                    transcript_text = "\n".join(f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}" for x in segments)
                    previous = [scene for batch in project["batches"] for scene in batch["scenes"]]
                    previous_context = [{"title": x["title"], "visual_prompt": x["visual_prompt"]} for x in previous[-6:]]
                    feedback = ""
                    for attempt in range(3):
                        try:
                            scenes = make_scene_plan(client, transcript_text, offset, batch_duration, planner_model,
                                scene_min, scene_max, max_scenes, cm_mode, language=language, enable_rich=enable_rich_overlay,
                                char_lock=char_lock, enable_sfx=enable_sfx, enable_music=enable_music, user_script=batch_script,
                                use_script_mode=internal_mode, style_mode=style_mode, previous_scenes=previous_context, feedback=feedback, output_budget=planner_output_budget)
                            batch = {"offset": offset, "duration": batch_duration, "language": language,
                                     "segments": segments, "scenes": scenes, "chunk": chunk.name}
                            validate_storyboard(project["batches"] + [batch]); break
                        except PlannerRateLimit:
                            raise
                        except (ValueError, RuntimeError) as exc:
                            if attempt == 2: raise
                            feedback = str(exc)[:400]
                            st.warning(f"Đang sửa kịch bản (lần {attempt+2}/3): {feedback}")
                    project["batches"].append(batch); write_json(plan_path, project)
                    offset += batch_duration
                project["complete"] = True; write_json(plan_path, project)
                captions = srt_text([dict(segment, start=batch["offset"]+max(0, segment["start"]),
                                     end=batch["offset"]+min(batch["duration"], segment["end"]))
                                    for batch in project["batches"] for segment in batch["segments"]])
                (root / "subtitles.srt").write_text(captions, encoding="utf-8")
                if internal_mode == "text_only": st.info("Phụ đề chế độ chỉ text là thời gian ước lượng theo độ dài văn bản.")
                if start:
                    validate_storyboard(project["batches"])
                    output_key = fingerprint([project, export_size, normalize_voice, burn_subtitles, allow_placeholder])
                    final = root / "video_final.mp4"
                    if not artifact_ok(final, output_key):
                        batch_videos = []
                        for index, batch in enumerate(project["batches"]):
                            st.markdown(f"### 🎬 Tạo video đợt {index+1}/{len(project['batches'])}")
                            work = root / f"work_{index+1:03d}"; work.mkdir(exist_ok=True)
                            batch_video = render_batch(chunks_dir / batch["chunk"], batch["scenes"], work,
                                Path(__file__).with_name("hand.png"), draw_style, cf_account, cf_token, hf_token,
                                freetheai_key, together_key, nexa_key, agnes_key, pollinations_key, pollinations_model,
                                image_timeout, flux_steps, fair_share_enabled, circuit_breaker_enabled, prioritize_fast,
                                char_lock=char_lock, enable_arrows=enable_arrows, enable_shadow=enable_shadow,
                                enable_sfx=enable_sfx, sfx_vol=sfx_volume, enable_music=enable_music, music_vol=music_volume,
                                seed_lock=project["seed"], style_mode=style_mode, language=batch["language"],
                                allow_placeholder=allow_placeholder, consistent_provider=consistent_provider)
                            batch_videos.append(batch_video)
                        with st.spinner("Đang hoàn thiện âm thanh và xuất video..."):
                            final = finish_export(batch_videos, root, duration, captions)
                        mark_artifact(final, output_key)
                    st.success("Video đã hoàn thành. Có thể tải bên dưới.")
                else: st.success("Kịch bản đã sẵn sàng. Xem và sửa bên dưới trước khi tạo ảnh.")
        except Exception as exc:
            st.error(f"Chưa hoàn tất: {exc}")
            if isinstance(exc, PlannerRateLimit) and exc.details:
                with st.expander("Chi tiết lỗi gốc Groq (Limit / Used / Requested)"):
                    st.code(exc.details, language="text")
            st.info("Các bước hoàn thành đã lưu. Khắc phục lỗi rồi bấm BẮT ĐẦU / TIẾP TỤC trong phiên này.")

    project = read_json(plan_path)
    if project and project.get("batches"):
        rows = [{"batch": bi+1, "scene": si+1, "start": round(batch["offset"]+scene["start"], 2),
                 "end": round(batch["offset"]+scene["end"], 2), "title": scene["title"],
                 "visual_prompt": scene["visual_prompt"], "callout_text": scene.get("callout_text", "")}
                for bi, batch in enumerate(project["batches"]) for si, scene in enumerate(batch["scenes"])]
        with st.expander(f"📝 Kịch bản • {len(rows)} cảnh — sửa tiêu đề, lời trên ảnh và mô tả", expanded=plan_only):
            st.caption("Mốc thời gian được khóa để giữ khớp voice. Mô tả ảnh nên viết bằng tiếng Anh.")
            with st.form(f"editor_{project_id}"):
                edited = st.data_editor(rows, disabled=["batch", "scene", "start", "end"],
                    use_container_width=True, hide_index=True, num_rows="fixed")
                save_edits = st.form_submit_button("💾 Lưu chỉnh sửa kịch bản")
            if save_edits:
                try:
                    with job_lock(root):
                        changed = json.loads(json.dumps(project))
                        for row in edited:
                            scene = changed["batches"][int(row["batch"])-1]["scenes"][int(row["scene"])-1]
                            for field in ("title", "visual_prompt", "callout_text"): scene[field] = str(row[field] or "").strip()
                        validate_storyboard(changed["batches"])
                        write_json(plan_path, changed); project = changed
                    st.success("Đã lưu. Bấm BẮT ĐẦU để cập nhật các cảnh đã sửa.")
                except Exception as exc: st.error(str(exc))
            st.download_button("⬇️ Kịch bản JSON", json.dumps(project, ensure_ascii=False, indent=2),
                file_name="storyboard.json", mime="application/json", on_click="ignore")
        with st.container(border=True):
            st.markdown("### 🎞️ Xem ảnh và nội dung từng cảnh")
            selected = st.slider("Kéo để xem cảnh", 1, len(rows), 1, key=f"scene_preview_{project_id}") if len(rows)>1 else 1
            row = rows[selected-1]
            batch = project["batches"][row["batch"]-1]
            scene = batch["scenes"][row["scene"]-1]
            work = root / f"work_{row['batch']:03d}"
            picture = work / f"scene_{row['scene']:03d}.jpg"
            clip = work / f"scene_{row['scene']:03d}.mp4"
            image_key = scene_image_key(scene, enable_arrows, enable_shadow, style_mode, batch["language"])
            ready = artifact_ok(picture, image_key)
            st.markdown(f"**Cảnh {selected}/{len(rows)} — {scene['title']}**")
            st.caption(f"{row['start']:.2f}s → {row['end']:.2f}s • Đợt {row['batch']}, cảnh {row['scene']}")
            left, right = st.columns([3, 2])
            with left:
                if ready:
                    st.image(str(picture), caption="Ảnh đã tạo cho phiên bản cảnh hiện tại", use_container_width=True)
                elif picture.exists():
                    st.info("Ảnh đang là bản cũ hoặc ảnh dự phòng. Bấm tiếp tục để cập nhật cảnh này.")
                else:
                    st.info("Chưa tạo ảnh cho cảnh này. Bạn có thể kiểm tra nội dung trước.")
            with right:
                st.markdown("**Lời đọc liên quan**")
                st.write(scene_narration(scene, batch["segments"]) or "Không có lời đọc trong khoảng này.")
                st.caption("Lời đọc hiển thị theo đoạn Whisper giao với cảnh; ranh giới câu có thể cần kiểm tra bằng tai.")
                st.markdown("**Mô tả gửi AI tạo ảnh**")
                st.write(scene["visual_prompt"])
                st.caption(f"Camera: {scene.get('camera_motion','auto')} • SFX: {scene.get('sfx','none')} • Nhạc: {scene.get('music_emotion','none')}")
            with st.expander("▶️ Nghe voice và xem đoạn diễn hoạt"):
                audio_chunk = root / "batches" / batch["chunk"]
                if audio_chunk.exists():
                    st.audio(str(audio_chunk), start_time=max(0, int(scene["start"])))
                    st.caption("Voice bắt đầu gần đầu cảnh và tiếp tục tới cuối đợt; dừng nghe ở mốc kết thúc cảnh bên trên.")
                durations = frame_durations(batch["scenes"], FPS)
                hand = Path(__file__).with_name("hand.png")
                clip_key = fingerprint({"version":VERSION,"image":file_digest(picture),"duration":durations[row["scene"]-1],
                    "style":draw_style,"motion":scene.get("camera_motion","zoom_in_center"),"mode":style_mode,
                    "language":batch["language"],"hand":file_digest(hand) if hand.exists() else "fallback"}) if ready else None
                if ready and artifact_ok(clip, clip_key):
                    st.video(str(clip)); st.caption("Đoạn diễn hoạt trước khi trộn âm thanh.")
            with st.expander("🧩 Bảng ảnh tổng quan"):
                page_count = max(1, math.ceil(len(rows)/6))
                page = st.number_input("Trang ảnh", min_value=1, max_value=page_count, value=1, step=1,
                                       key=f"scene_page_{project_id}")
                columns = st.columns(2)
                for position, item in enumerate(rows[(page-1)*6:page*6]):
                    other_batch = project["batches"][item["batch"]-1]
                    other_scene = other_batch["scenes"][item["scene"]-1]
                    other = root / f"work_{item['batch']:03d}" / f"scene_{item['scene']:03d}.jpg"
                    with columns[position%2]:
                        st.caption(f"{item['start']:.1f}–{item['end']:.1f}s • {other_scene['title']}")
                        if artifact_ok(other, scene_image_key(other_scene, enable_arrows, enable_shadow, style_mode, other_batch["language"])):
                            st.image(str(other), use_container_width=True)
                        else: st.caption("Chưa có ảnh hợp lệ cho phiên bản này.")
        alignment = read_json(root / "alignment.json")
        if alignment:
            with st.expander("🔎 Đối chiếu text và lời nhận dạng"):
                comparison = [{"đợt":i+1,"bắt đầu":segment["start"],"kết thúc":segment["end"],
                               "Whisper":segment.get("original_text",segment["text"]),"Text đã căn":segment["text"],
                               "Cần kiểm tra":segment.get("alignment_review",False)}
                              for i, batch in enumerate(alignment["batches"]) for segment in batch["segments"]]
                st.dataframe(comparison, use_container_width=True, hide_index=True)
        subtitles = root / "subtitles.srt"
        if subtitles.exists():
            st.download_button("⬇️ Phụ đề SRT", subtitles.read_bytes(), file_name="subtitles.srt", on_click="ignore")
        final = root / "video_final.mp4"
        output_key = fingerprint([project, export_size, normalize_voice, burn_subtitles, allow_placeholder])
        if artifact_ok(final, output_key):
            st.video(str(final))
            st.download_button("⬇️ TẢI VIDEO", final.read_bytes(), file_name="video_final.mp4",
                               mime="video/mp4", use_container_width=True, on_click="ignore")
