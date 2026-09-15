"""
Xưởng Video Diễn Hoạt Kiến Thức AI — Bản Siêu Cấp V5.4
========================================================
Tổng hợp tất cả fix và tính năng:
1. Fix lỗi break fallback (cảnh không bị mất)
2. Fair share cap (không provider nào chiếm quá nhiều)
3. Live Dashboard real-time (scene status + provider stats)
4. Log lỗi chi tiết từng provider
5. 4 hàm render riêng (kttv_v2, hybrid, pure, classic_hand)
6. Provider 1-lần-thử (không retry 2 tầng)
7. Font fallback 4 lớp (system → Pillow → Internet → default)
8. Title band 95px cố định
9. 8 camera motion + 3 chế độ chọn (Auto/Random/Cố định)
10. Ảnh placeholder nếu mọi provider fail
"""

import os
import re
import io
import json
import math
import time
import base64
import random
import shutil
import subprocess
import tempfile
import threading
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
# CẤU HÌNH CHUNG
# ============================================================
APP_TITLE = "Xưởng Video Diễn Hoạt Kiến Thức AI (Bản Siêu Cấp V5.4)"
BATCH_SECONDS = 10 * 60
FPS = 30
WIDTH = 1280
HEIGHT = 720
TITLE_BAND_H = 95
CONTENT_H = HEIGHT - TITLE_BAND_H
REVEAL_RADIUS = 34
DRAW_DURATION_RATIO = 0.55
PHASE_RATIOS = (0.40, 0.35, 0.25)
MAX_TOTAL_FALLBACK_TIME = 300
FAIR_SHARE_MULTIPLIER = 1.5  # Mỗi provider tối đa 1.5x share trung bình

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

# ============================================================
# GIAO DIỆN
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title("🎬 Xưởng Video Diễn Hoạt Kiến Thức AI — Siêu Cấp V5.4")
st.caption("Live Dashboard + Fair Share Cap + 4 hàm render riêng + Font fallback 4 lớp")

with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password")

    with st.expander("🎨 Nhà cung cấp ảnh AI", expanded=True):
        pollinations_key = st.text_input("Pollinations API Key",
            value=st.secrets.get("POLLINATIONS_API_KEY", os.getenv("POLLINATIONS_API_KEY", "")),
            type="password")
        pollinations_model = st.selectbox("Pollinations Model",
            ["flux-pro", "flux", "gptimage", "kontext", "flux-realism"], index=0)
        agnes_key = st.text_input("Agnes AI API Key (Miễn phí)",
            value=st.secrets.get("AGNES_API_KEY", os.getenv("AGNES_API_KEY", "")),
            type="password")
        cf_account = st.text_input("Cloudflare Account ID",
            value=st.secrets.get("CLOUDFLARE_ACCOUNT_ID", os.getenv("CLOUDFLARE_ACCOUNT_ID", "")),
            type="password")
        cf_token = st.text_input("Cloudflare API Token",
            value=st.secrets.get("CLOUDFLARE_API_TOKEN", os.getenv("CLOUDFLARE_API_TOKEN", "")),
            type="password")
        hf_token = st.text_input("Hugging Face Token (Miễn phí)",
            value=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "")),
            type="password")
        freetheai_key = st.text_input("FreeTheAi API Key (Miễn phí)",
            value=st.secrets.get("FREETHEAI_API_KEY", os.getenv("FREETHEAI_API_KEY", "")),
            type="password")
        together_key = st.text_input("Together AI API Key",
            value=st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY", "")),
            type="password")
        nexa_key = st.text_input("NexaAPI Key ($5 free)",
            value=st.secrets.get("NEXA_API_KEY", os.getenv("NEXA_API_KEY", "")),
            type="password")

    st.header("🧠 Mô hình Groq")
    stt_model = st.selectbox("STT Model", ["whisper-large-v3", "whisper-large-v3-turbo"], index=0)
    planner_model = st.selectbox("Biên kịch Model",
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=0)

    st.header("🎬 Phong cách diễn hoạt")
    draw_style = st.selectbox("Chọn phong cách", [
        "1. Kiến Thức Thú Vị V2 (Vẽ tuần tự 3 phase + Camera Pan/Zoom)",
        "2. Độc bản Hybrid (Tay vẽ bám nét + Camera Steadicam)",
        "3. Chỉ Camera Pan & Zoom (ẩn bàn tay)",
        "4. Bảng trắng cổ điển (Tay vẽ góc máy tĩnh)",
    ], index=0)

    st.header("🎥 Camera Motion")
    camera_motion_mode = st.selectbox("Chế độ chuyển động camera", [
        "Auto (AI chọn cho từng cảnh)",
        "Random (Code chọn ngẫu nhiên)",
        "Cố định: zoom_in_center",
        "Cố định: zoom_out_center",
        "Cố định: pan_left_to_right",
        "Cố định: pan_right_to_left",
        "Cố định: ken_burns_slow",
        "Cố định: static",
    ], index=0)

    st.header("⏱️ Khóa nhịp cảnh")
    scene_min = st.slider("Tối thiểu (giây)", 18, 25, 19)
    scene_max = st.slider("Tối đa (giây)", 22, 35, 27)
    if scene_max < scene_min:
        scene_max = scene_min

    st.header("⚙️ Cài đặt khác")
    max_scenes = st.slider("Số cảnh tối đa mỗi batch", 5, 50, 25)
    image_timeout = st.slider("Timeout tạo ảnh (giây)", 20, 90, 45)
    flux_steps = st.slider("Số bước FLUX", 4, 8, 4)
    fair_share_enabled = st.checkbox("Bật fair share cap", value=True,
        help="Giới hạn mỗi provider tối đa 1.5x share trung bình để công bằng")

# ============================================================
# TIỆN ÍCH
# ============================================================
def run_cmd(cmd, timeout=600):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-5000:] or "Lệnh thất bại")
    return p.stdout

def ffprobe_duration(path):
    out = run_cmd(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60)
    return float(out.strip())

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [m.start() for m in re.finditer(r"\{", text)]
    ends = [m.end() for m in re.finditer(r"\}", text)]
    for s in starts:
        for e in reversed(ends):
            if e <= s:
                continue
            try:
                return json.loads(text[s:e])
            except Exception:
                continue
    raise ValueError("AI không phản hồi JSON hợp lệ")

def groq_client(key):
    return Groq(api_key=key)

def transcribe_file(client, path, model):
    with open(path, "rb") as f:
        return client.audio.transcriptions.create(
            file=(Path(path).name, f.read()), model=model,
            response_format="verbose_json", timestamp_granularities=["segment"],
            language="vi", temperature=0.0)

def chunk_audio(src, out_dir):
    pattern = str(Path(out_dir) / "batch_%03d.m4a")
    run_cmd(["ffmpeg", "-y", "-i", str(src), "-map", "0:a:0",
             "-c:a", "aac", "-b:a", "96k", "-f", "segment",
             "-segment_time", str(BATCH_SECONDS), "-reset_timestamps", "1", pattern], timeout=900)
    return sorted(Path(out_dir).glob("batch_*.m4a"))

def normalize_segments(result, offset):
    data = result.model_dump() if hasattr(result, "model_dump") else result
    segments = data.get("segments", []) if isinstance(data, dict) else getattr(result, "segments", [])
    out = []
    for s in segments or []:
        if isinstance(s, dict):
            stt = float(s.get("start", 0)); end = float(s.get("end", stt)); text = str(s.get("text", "")).strip()
        else:
            stt = float(getattr(s, "start", 0)); end = float(getattr(s, "end", stt)); text = str(getattr(s, "text", "")).strip()
        if text:
            out.append({"start": stt + offset, "end": end + offset, "text": text})
    return out

def sanitize_prompt_text(prompt):
    for pattern, rep in {
        r"\bblood\b": "dark ink", r"\bbleed\b": "drip", r"\bsuicide\b": "despair",
        r"\bkill(ing|er)?\b": "oppression", r"\bdead\b": "fallen", r"\bdeath\b": "crisis",
        r"\bcorpse\b": "shadow", r"\bjump(ing)?\b": "falling shadow", r"\bweapon\b": "heavy chain",
    }.items():
        prompt = re.sub(pattern, rep, prompt, flags=re.IGNORECASE)
    return prompt

# ============================================================
# TITLE BAND HELPERS
# ============================================================
def split_title_band(image_bgr):
    return image_bgr[:TITLE_BAND_H, :].copy(), image_bgr[TITLE_BAND_H:, :].copy()

def compose_frame(title_band_bgr, content_bgr):
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:TITLE_BAND_H] = title_band_bgr
    canvas[TITLE_BAND_H:] = content_bgr
    return canvas

def crop_content_with_motion(content_bgr, scale, cx, cy):
    ch, cw = content_bgr.shape[:2]
    crop_w = max(1, min(cw, int(cw / max(0.5, scale))))
    crop_h = max(1, min(ch, int(ch / max(0.5, scale))))
    x1 = max(0, min(cw - crop_w, int(cx - crop_w / 2)))
    y1 = max(0, min(ch - crop_h, int(cy - crop_h / 2)))
    return cv2.resize(content_bgr[y1:y1+crop_h, x1:x1+crop_w], (cw, ch), interpolation=cv2.INTER_LINEAR)

def trajectory_to_content_space(trajectory_full):
    out = [(int(px), int(py - TITLE_BAND_H)) for (px, py) in trajectory_full if py >= TITLE_BAND_H]
    return out if out else [(WIDTH // 2, CONTENT_H // 2)]

def split_trajectory_into_phases(trajectory, ratios=PHASE_RATIOS):
    n = len(trajectory)
    if n < 3:
        return [trajectory, [], []]
    b1 = max(1, int(n * ratios[0]))
    b2 = max(b1 + 1, int(n * (ratios[0] + ratios[1])))
    b2 = min(b2, n - 1)
    return [trajectory[:b1], trajectory[b1:b2], trajectory[b2:]]

# ============================================================
# FONT FALLBACK 4 LỚP
# ============================================================
@st.cache_resource(show_spinner=False)
def _download_fallback_font():
    cache_path = Path("/tmp/DejaVuSans-Bold.ttf")
    if cache_path.exists():
        return str(cache_path)
    for url in [
        "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
        "https://cdn.jsdelivr.net/gh/dejavu-fonts/dejavu-fonts/ttf/DejaVuSans-Bold.ttf",
    ]:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100000:
                cache_path.write_bytes(r.content)
                return str(cache_path)
        except Exception:
            continue
    return None

def font_for(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    for fb in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(fb, size)
        except Exception:
            continue
    try:
        cached = _download_fallback_font()
        if cached:
            return ImageFont.truetype(cached, size)
    except Exception:
        pass
    return ImageFont.load_default()

# ============================================================
# SCENE PLANNER
# ============================================================
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model,
                    min_s, max_s, max_scenes, camera_mode="auto"):
    expected_scenes = max(1, round(batch_duration / 23.0))
    system = f"""
Bạn là giám đốc sáng tạo kịch bản cho kênh hoạt họa kiến thức phong cách "Kiến Thức Thú Vị".
Nhiệm vụ: Chia đoạn âm thanh {batch_duration:.0f}s thành khoảng {expected_scenes} cảnh lớn ({min_s}-{max_s}s/cảnh).

QUY TẮC VỀ TIÊU ĐỀ ("title"):
- Tiếng Việt tự nhiên, 3-6 từ, VIẾT HOA, tóm tắt luận điểm chính.
- KHÔNG dùng từ ghép kiểu dịch máy.
- Ví dụ: "CỐ GẮNG HÀI LÒNG MỌI NGƯỜI", "NỖI SỢ BỊ PHÁN XÉT".

QUY TẮC VỀ CHỮ TRÊN TRANH ("callout_type", "callout_text"):
- "speech": bong bóng thoại. "thought": đám mây suy nghĩ. "sticker": nhãn dán. "none": không chữ.
- LUÂN PHIÊN thay đổi.

QUY TẮC QUAN TRỌNG NHẤT — MÔ TẢ TRANH ("visual_prompt") PHẢI SÁNG TẠO:
MỖI CẢNH LÀ MỘT "SÂN KHẤU" KHÁC NHAU. TUYỆT ĐỐI KHÔNG lặp bố cục.
Bao gồm: nhân vật + tư thế, hành động, bối cảnh, đồ vật ẩn dụ, cảm xúc, màu nhấn.
VÍ DỤ TỐT:
- "2D comic doodle: a young man standing at the edge of a cliff at sunset, looking down at a vast ocean of papers below, red sunset, blue waves, white background, bold black outlines, no text"
- "2D comic doodle: a man trapped inside a giant glass jar, hands pressing against walls, red accents, white background, no text"
VÍ DỤ XẤU: "a man sitting at a desk with papers on the left".

RÀNG BUỘC PHONG CÁCH: 2D comic doodle, nét mực đen dày, nền TRẮNG TINH, KHÔNG chữ/số/bong bóng rỗng.

QUY TẮC CAMERA MOTION ("camera_motion"):
Chọn 1 trong: "zoom_in_center", "zoom_out_center", "pan_left_to_right", "pan_right_to_left",
"zoom_in_top_left", "zoom_in_bottom_right", "ken_burns_slow", "static". LUÂN PHIÊN.

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0, "end": 22.0,
      "title": "NỖI SỢ BỊ PHÁN XÉT",
      "callout_type": "thought", "callout_text": "TỚ ĐANG NGHĨ GÌ?", "callout_side": "right",
      "camera_motion": "zoom_in_center",
      "visual_prompt": "2D comic doodle: ..."
    }}
  ]
}}
"""
    user = f"Độ dài âm thanh: {batch_duration:.2f}s.\nCHỈ CHIA TỐI ĐA {expected_scenes} CẢNH ({min_s}-{max_s}s/cảnh).\n\nTRANSCRIPT:\n{transcript_text}"

    try:
        r = client.chat.completions.create(
            model=model, temperature=0.15, max_tokens=16000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        obj = extract_json(r.choices[0].message.content)
        raw_scenes = obj.get("scenes", [])
    except Exception:
        raw_scenes = []

    valid_motions = {"zoom_in_center", "zoom_out_center", "pan_left_to_right", "pan_right_to_left",
                     "zoom_in_top_left", "zoom_in_bottom_right", "ken_burns_slow", "static"}

    clean = []
    for s in raw_scenes[:max_scenes]:
        try:
            a = max(0.0, float(s["start"])); b = min(batch_duration, float(s["end"]))
            if b <= a + 1.0:
                continue
            vp = sanitize_prompt_text(str(s.get("visual_prompt", "")))
            if not vp or len(vp) < 10:
                continue
            ct = str(s.get("callout_type", "speech")).strip().lower()
            if ct not in ("speech", "thought", "sticker", "none"):
                ct = "speech"
            cm = str(s.get("camera_motion", "zoom_in_center")).strip().lower()
            if cm not in valid_motions:
                cm = "zoom_in_center"
            clean.append({"start": a, "end": b,
                "title": str(s.get("title", "BÀI HỌC KIẾN THỨC")).strip().upper(),
                "callout_type": ct, "callout_text": str(s.get("callout_text", "")).strip(),
                "callout_side": str(s.get("callout_side", "right")).strip().lower(),
                "camera_motion": cm, "visual_prompt": vp})
        except Exception:
            continue

    if not clean:
        clean = [{"start": 0.0, "end": batch_duration, "title": "BÀI HỌC QUAN TRỌNG",
            "callout_type": "speech", "callout_text": "RẤT SAI LẦM!", "callout_side": "right",
            "camera_motion": "zoom_in_center",
            "visual_prompt": "2D comic doodle: a man standing at the edge of a cliff at sunset, red sunset, blue waves, white background, bold black outlines, no text"}]

    merged = []
    for s in clean:
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            if (s["end"] - s["start"]) < 17.0 or (s["start"] - prev["start"] < 17.0):
                prev["end"] = max(prev["end"], s["end"])
                if not prev.get("callout_text") and s.get("callout_text"):
                    prev["callout_text"] = s["callout_text"]; prev["callout_type"] = s["callout_type"]
            else:
                merged.append(s)
    clean = merged
    clean[0]["start"] = 0.0
    for i in range(len(clean) - 1):
        clean[i]["end"] = clean[i + 1]["start"]
    clean[-1]["end"] = batch_duration

    final_scenes = []
    for s in clean:
        dur = s["end"] - s["start"]
        if dur > 35.0:
            mid = s["start"] + dur / 2.0
            final_scenes.append({"start": s["start"], "end": mid, "title": s["title"],
                "callout_type": s.get("callout_type", "speech"), "callout_text": s.get("callout_text", ""),
                "callout_side": s.get("callout_side", "right"), "camera_motion": s.get("camera_motion", "zoom_in_center"),
                "visual_prompt": s["visual_prompt"]})
            final_scenes.append({"start": mid, "end": s["end"], "title": f"{s['title']} (TIẾP)",
                "callout_type": "sticker", "callout_text": "CẦN CẨN TRỌNG!", "callout_side": "right",
                "camera_motion": "zoom_out_center",
                "visual_prompt": s["visual_prompt"] + ", continuation scene, different angle, clean white background"})
        else:
            final_scenes.append(s)

    if camera_mode == "random":
        motions_list = list(valid_motions - {"static"})
        for s in final_scenes:
            s["camera_motion"] = random.choice(motions_list)
    elif camera_mode.startswith("fixed:"):
        fixed = camera_mode.split(":", 1)[1].strip()
        if fixed in valid_motions:
            for s in final_scenes:
                s["camera_motion"] = fixed
    return final_scenes

# ============================================================
# IMAGE PROVIDERS (1 LẦN THỬ)
# ============================================================
def _build_full_prompt(prompt):
    safe = sanitize_prompt_text(prompt)
    return f"""{safe}.

STYLE CONSTRAINTS (chỉ về phong cách vẽ, KHÔNG áp bố cục):
- Authentic 2D comic doodle art style, thick black ink contour outlines, hand-drawn wobbly lines.
- Pure solid flat white background OR simple scene background if described above.
- Vivid expressive cartoon character with clear emotion.
- Selective vibrant spot colors (red, blue, orange, green) ONLY on key symbolic elements.
- Absolutely NO text, letters, numbers, captions, or empty speech balloons.
- Do not draw desk, table, markers, pens UNLESS explicitly mentioned.
- Wide 16:9 cinematic composition.
"""

def _validate_image_bytes(data, provider_name):
    if not data or len(data) < 500:
        raise RuntimeError(f"{provider_name}: dữ liệu quá nhỏ")
    if not (data[:3] == b'\xff\xd8\xff' or data[:8] == b'\x89PNG\r\n\x1a\n' or data[:4] == b'RIFF'):
        preview = data[:150].decode("utf-8", errors="ignore").lower()
        if "<html" in preview or "<!doctype" in preview:
            raise RuntimeError(f"{provider_name}: HTML response")
        raise RuntimeError(f"{provider_name}: không phải ảnh")
    return data

def agnes_image_request(prompt, api_key, timeout=45):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Agnes: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": AGNES_MODEL, "prompt": _build_full_prompt(prompt), "size": "1280x720",
               "extra_body": {"response_format": "b64_json"}}
    r = requests.post(AGNES_API_URL, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("Agnes: rate limit (429)")
    if r.status_code == 401:
        raise RuntimeError("Agnes: key sai (401)")
    if r.status_code >= 400:
        raise RuntimeError(f"Agnes HTTP {r.status_code}")
    data = r.json()
    item = data.get("data", [{}])[0]
    if item.get("b64_json"):
        return _validate_image_bytes(base64.b64decode(item["b64_json"]), "Agnes")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate_image_bytes(img.content, "Agnes")
    raise RuntimeError("Agnes: no image")

def cloudflare_image_request(prompt, account_id, api_token, timeout=45, steps=4):
    account_id = (account_id or "").strip(); api_token = (api_token or "").strip()
    if not account_id or not api_token:
        raise RuntimeError("Cloudflare: thiếu thông tin")
    url = f"{CLOUDFLARE_BASE}{account_id}/ai/run/{CLOUDFLARE_MODEL}"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers,
                      json={"prompt": _build_full_prompt(prompt), "steps": steps}, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("Cloudflare: hết quota (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"Cloudflare HTTP {r.status_code}")
    data = r.json()
    if not data.get("success", True):
        raise RuntimeError("Cloudflare fail")
    b64 = data.get("result", {}).get("image")
    if not b64:
        raise RuntimeError("Cloudflare: no image")
    return _validate_image_bytes(base64.b64decode(b64), "Cloudflare")

def hf_image_request(prompt, token, timeout=45):
    token = (token or "").strip()
    if not token:
        raise RuntimeError("HF: chưa có token")
    url = f"{HF_API_URL}{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers,
                      json={"inputs": _build_full_prompt(prompt)}, timeout=timeout)
    if r.status_code == 503:
        raise RuntimeError("HF: model loading (503)")
    if r.status_code == 429:
        raise RuntimeError("HF: rate limit (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"HF HTTP {r.status_code}")
    return _validate_image_bytes(r.content, "HF")

def freetheai_image_request(prompt, api_key, timeout=45):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("FreeTheAi: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "flux", "prompt": _build_full_prompt(prompt), "n": 1, "size": "1280x720"}
    r = requests.post(FREETHEAI_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("FreeTheAi: rate limit (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"FreeTheAi HTTP {r.status_code}")
    data = r.json()
    item = data.get("data", [{}])[0]
    if item.get("b64_json"):
        return _validate_image_bytes(base64.b64decode(item["b64_json"]), "FreeTheAi")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate_image_bytes(img.content, "FreeTheAi")
    raise RuntimeError("FreeTheAi: no image")

def together_image_request(prompt, api_key, timeout=45):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Together: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": TOGETHER_MODEL, "prompt": _build_full_prompt(prompt),
               "width": WIDTH, "height": HEIGHT, "steps": 4, "n": 1, "response_format": "b64_json"}
    r = requests.post(TOGETHER_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("Together: rate limit (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"Together HTTP {r.status_code}")
    data = r.json()
    item = data.get("data", [{}])[0]
    if item.get("b64_json"):
        return _validate_image_bytes(base64.b64decode(item["b64_json"]), "Together")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate_image_bytes(img.content, "Together")
    raise RuntimeError("Together: no image")

def nexa_image_request(prompt, api_key, timeout=45):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("NexaAPI: chưa có key")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": NEXA_MODEL, "prompt": _build_full_prompt(prompt),
               "width": WIDTH, "height": HEIGHT, "n": 1}
    r = requests.post(NEXA_BASE, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("NexaAPI: rate limit (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"NexaAPI HTTP {r.status_code}")
    data = r.json()
    item = data.get("data", [{}])[0]
    if item.get("b64_json"):
        return _validate_image_bytes(base64.b64decode(item["b64_json"]), "NexaAPI")
    if item.get("url"):
        img = requests.get(item["url"], timeout=timeout)
        return _validate_image_bytes(img.content, "NexaAPI")
    raise RuntimeError("NexaAPI: no image")

def pollinations_image_request(prompt, api_key, model="flux-pro", timeout=45, seed=None):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Pollinations: chưa có key")
    encoded = requests.utils.quote(_build_full_prompt(prompt), safe="")
    url = f"{POLLINATIONS_BASE}{encoded}"
    params = {"width": WIDTH, "height": HEIGHT, "model": model,
              "nologo": "true", "enhance": "true", "safe": "false"}
    if seed is not None:
        params["seed"] = seed
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=True)
    if r.status_code == 401:
        raise RuntimeError("Pollinations 401: key sai")
    if r.status_code == 402:
        raise RuntimeError("Pollinations 402: hết credit")
    if r.status_code == 429:
        raise RuntimeError("Pollinations 429: rate limit")
    if r.status_code >= 400:
        raise RuntimeError(f"Pollinations HTTP {r.status_code}")
    return _validate_image_bytes(r.content, "Pollinations")

def build_provider_list(cf_account, cf_token, hf_token, freetheai_key,
                         together_key, nexa_key, agnes_key,
                         pollinations_key="", pollinations_model="flux-pro",
                         flux_steps=4):
    providers = []
    if pollinations_key and pollinations_key.strip():
        providers.append({"name": f"Pollinations ({pollinations_model})",
                          "fn": pollinations_image_request,
                          "args": [pollinations_key.strip(), pollinations_model],
                          "kwargs": {}, "supports_seed": ["seed"]})
    if agnes_key and agnes_key.strip():
        providers.append({"name": "Agnes AI", "fn": agnes_image_request, "args": [agnes_key.strip()]})
    if cf_account and cf_token and cf_account.strip() and cf_token.strip():
        providers.append({"name": "Cloudflare", "fn": cloudflare_image_request,
                          "args": [cf_account.strip(), cf_token.strip()],
                          "kwargs": {"steps": flux_steps}})
    if hf_token and hf_token.strip():
        providers.append({"name": "Hugging Face", "fn": hf_image_request, "args": [hf_token.strip()]})
    if freetheai_key and freetheai_key.strip():
        providers.append({"name": "FreeTheAi", "fn": freetheai_image_request, "args": [freetheai_key.strip()]})
    if together_key and together_key.strip():
        providers.append({"name": "Together AI", "fn": together_image_request, "args": [together_key.strip()]})
    if nexa_key and nexa_key.strip():
        providers.append({"name": "NexaAPI", "fn": nexa_image_request, "args": [nexa_key.strip()]})
    return providers

def save_image_from_bytes(data, output_path):
    if not data:
        raise RuntimeError("Dữ liệu ảnh rỗng")
    Path(output_path).write_bytes(data)
    try:
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            im = im.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            im.save(output_path, "JPEG", quality=95)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Ảnh không hợp lệ: {e}")

# ============================================================
# OVERLAY COMIC
# ============================================================
def add_comic_overlays(image_path, title, callout_type, callout_text, callout_side, output_path):
    img = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    if title:
        draw.rectangle([0, 0, WIDTH, TITLE_BAND_H], fill="white")
        f_size = 36
        f_title = font_for(f_size)
        while f_size > 18:
            box = draw.textbbox((0, 0), title, font=f_title)
            if box[2] - box[0] <= 850:
                break
            f_size -= 2
            f_title = font_for(f_size)
        box = draw.textbbox((0, 0), title, font=f_title)
        tw = box[2] - box[0]
        draw.text(((WIDTH - tw) / 2, 26), title, fill="black", font=f_title)

    if callout_text and callout_type != "none":
        f_text = font_for(25)
        bb = draw.textbbox((0, 0), callout_text, font=f_text)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        cx, cy = (int(WIDTH * 0.28), int(HEIGHT * 0.40)) if callout_side == "left" else (int(WIDTH * 0.74), int(HEIGHT * 0.38))

        if callout_type == "speech":
            pad_x, pad_y = 18, 12
            rect = [cx - bw // 2 - pad_x, cy - bh // 2 - pad_y, cx + bw // 2 + pad_x, cy + bh // 2 + pad_y]
            draw.rounded_rectangle(rect, radius=14, fill="white", outline="black", width=4)
            tail_tip = (cx - 20, cy + bh // 2 + pad_y + 18)
            draw.polygon([(cx - 32, cy + bh // 2 + pad_y - 2), (cx - 8, cy + bh // 2 + pad_y - 2), tail_tip],
                         fill="white", outline="black")
            draw.line([(cx - 30, cy + bh // 2 + pad_y), (cx - 10, cy + bh // 2 + pad_y)], fill="white", width=5)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill="#1b5e20", font=f_text)
        elif callout_type == "thought":
            pad_x, pad_y = 22, 14
            rect = [cx - bw // 2 - pad_x, cy - bh // 2 - pad_y, cx + bw // 2 + pad_x, cy + bh // 2 + pad_y]
            draw.rounded_rectangle(rect, radius=24, fill="white", outline="black", width=3)
            draw.ellipse([cx - 20, cy + bh // 2 + pad_y + 4, cx - 10, cy + bh // 2 + pad_y + 14], fill="white", outline="black", width=3)
            draw.ellipse([cx - 28, cy + bh // 2 + pad_y + 17, cx - 22, cy + bh // 2 + pad_y + 23], fill="white", outline="black", width=2)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill="#0d47a1", font=f_text)
        else:
            pad_x, pad_y = 16, 9
            bx, by = int(WIDTH * 0.75), int(HEIGHT * 0.88)
            b_rect = [bx - bw // 2 - pad_x, by - bh // 2 - pad_y, bx + bw // 2 + pad_x, by + bh // 2 + pad_y]
            draw.rounded_rectangle(b_rect, radius=8, fill="white", outline="#b71c1c", width=4)
            draw.text((bx - bw // 2, by - bh // 2 - 2), callout_text, fill="#b71c1c", font=f_text)
    img.save(output_path, quality=95)

# ============================================================
# HAND ASSET
# ============================================================
def generate_fallback_hand():
    S = 320
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([(40, 40), (140, 140)], fill=(30, 30, 30, 255), width=10)
    d.polygon([(30, 30), (50, 40), (40, 50)], fill=(220, 50, 50, 255))
    d.ellipse((110, 110, 260, 260), fill=(235, 205, 175, 255), outline=(30, 30, 30, 255), width=4)
    d.rounded_rectangle((100, 130, 180, 220), 20, fill=(235, 205, 175, 255), outline=(30, 30, 30, 255), width=4)
    return im

def load_hand_asset(hand_path, target_width=320):
    p = None
    for c in [hand_path, Path("hand.png"), Path("assets/hand.png")]:
        if c and Path(c).exists():
            p = Path(c); break
    pil_hand = Image.open(p).convert("RGBA") if p else generate_fallback_hand()
    w, h = pil_hand.size
    pil_hand = pil_hand.resize((target_width, int(h * target_width / w)), Image.Resampling.LANCZOS)
    hand_np = np.array(pil_hand)
    bgr = cv2.cvtColor(hand_np[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = hand_np[:, :, 3]
    alpha[:6, :] = 0; alpha[-6:, :] = 0; alpha[:, :6] = 0; alpha[:, -6:] = 0
    alpha[alpha < 110] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 50).astype(np.uint8))
    if num_labels > 1:
        alpha[labels != (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.erode(alpha, kernel, iterations=1)
    alpha = np.minimum(alpha, cv2.GaussianBlur(alpha, (3, 3), 0))
    ys, xs = np.where(alpha > 120)
    tip_x, tip_y = (int(xs[np.argmin(xs + ys * 1.15)]), int(ys[np.argmin(xs + ys * 1.15)])) if len(xs) > 0 else (0, 0)
    return bgr, alpha, tip_x, tip_y

def sort_contours_nn(contours, start_pt=(100, 150)):
    def c_center(c):
        M = cv2.moments(c)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x, y, w, h = cv2.boundingRect(c); return (x + w // 2, y + h // 2)
    valid = [c for c in contours if cv2.arcLength(c, False) > 10]
    sorted_res = []
    if valid:
        curr = start_pt; rem = valid[:]
        while rem:
            best_idx = 0; best_dist = float("inf")
            for idx, c in enumerate(rem):
                pt = c_center(c)
                d = (pt[0] - curr[0]) ** 2 + (pt[1] - curr[1]) ** 2
                if d < best_dist:
                    best_dist = d; best_idx = idx
            chosen = rem.pop(best_idx); sorted_res.append(chosen); curr = c_center(chosen)
    return sorted_res

def extract_continuous_trajectory(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return [(WIDTH // 2, HEIGHT // 2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    tm = np.zeros_like(binary); tm[:TITLE_BAND_H, :] = binary[:TITLE_BAND_H, :]
    bm = np.zeros_like(binary); bm[TITLE_BAND_H:, :] = binary[TITLE_BAND_H:, :]
    tc, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bc, _ = cv2.findContours(bm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ts = sorted(tc, key=lambda c: cv2.boundingRect(c)[0])
    bs = sort_contours_nn(bc, start_pt=(150, 200))
    trajectory = []
    for c in ts + bs:
        for p in c.reshape(-1, 2)[::3]:
            trajectory.append((int(p[0]), int(p[1])))
    if len(trajectory) < 40:
        trajectory = [(x, y) for y in range(120, 680, 45) for x in range(80, 1200, 25)]
    return trajectory

def extract_staggered_trajectories(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return [[(WIDTH // 2, HEIGHT // 2)]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    mt = np.zeros_like(binary); mt[:TITLE_BAND_H, :] = binary[:TITLE_BAND_H, :]
    ml = np.zeros_like(binary); ml[TITLE_BAND_H:HEIGHT, :int(WIDTH * 0.48)] = binary[TITLE_BAND_H:HEIGHT, :int(WIDTH * 0.48)]
    mr = np.zeros_like(binary); mr[TITLE_BAND_H:int(HEIGHT * 0.72), int(WIDTH * 0.48):] = binary[TITLE_BAND_H:int(HEIGHT * 0.72), int(WIDTH * 0.48):]
    mb = np.zeros_like(binary); mb[int(HEIGHT * 0.72):, int(WIDTH * 0.48):] = binary[int(HEIGHT * 0.72):, int(WIDTH * 0.48):]

    def to_pts(cnts):
        return [(int(p[0]), int(p[1])) for c in cnts for p in c.reshape(-1, 2)[::3]]

    tc, _ = cv2.findContours(mt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lc, _ = cv2.findContours(ml, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rc, _ = cv2.findContours(mr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    bc, _ = cv2.findContours(mb, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    zones = [
        to_pts(sorted(tc, key=lambda c: cv2.boundingRect(c)[0])),
        to_pts(sort_contours_nn(lc, start_pt=(150, 250))),
        to_pts(sort_contours_nn(rc, start_pt=(int(WIDTH * 0.7), 250))),
        to_pts(sort_contours_nn(bc, start_pt=(int(WIDTH * 0.75), int(HEIGHT * 0.85)))),
    ]
    return [z for z in zones if len(z) > 10]

def paste_hand(frame_bgr, hand_bgr, hand_alpha, x, y):
    fh, fw = frame_bgr.shape[:2]; hh, hw = hand_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y); x2, y2 = min(fw, x + hw), min(fh, y + hh)
    if x1 >= x2 or y1 >= y2:
        return
    hx1, hy1 = x1 - x, y1 - y; hx2, hy2 = hx1 + (x2 - x1), hy1 + (y2 - y1)
    sub_hand = hand_bgr[hy1:hy2, hx1:hx2]
    sub_alpha = (hand_alpha[hy1:hy2, hx1:hx2].astype(np.float32) / 255.0)[:, :, None]
    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    blended = sub_hand.astype(np.float32) * sub_alpha + roi * (1.0 - sub_alpha)
    frame_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

def ease_in_out(t):
    return 0.5 * (1.0 - math.cos(math.pi * t))

def get_motion_keyframes(motion):
    return {
        "zoom_in_center": [(0.0, 1.00, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.22, WIDTH*0.50, HEIGHT*0.50)],
        "zoom_out_center": [(0.0, 1.22, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.00, WIDTH*0.50, HEIGHT*0.50)],
        "pan_left_to_right": [(0.0, 1.12, WIDTH*0.35, HEIGHT*0.50), (1.0, 1.12, WIDTH*0.65, HEIGHT*0.50)],
        "pan_right_to_left": [(0.0, 1.12, WIDTH*0.65, HEIGHT*0.50), (1.0, 1.12, WIDTH*0.35, HEIGHT*0.50)],
        "zoom_in_top_left": [(0.0, 1.00, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.24, WIDTH*0.30, HEIGHT*0.35)],
        "zoom_in_bottom_right": [(0.0, 1.00, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.24, WIDTH*0.70, HEIGHT*0.65)],
        "ken_burns_slow": [(0.0, 1.05, WIDTH*0.45, HEIGHT*0.48), (1.0, 1.18, WIDTH*0.55, HEIGHT*0.52)],
        "static": [(0.0, 1.00, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.00, WIDTH*0.50, HEIGHT*0.50)],
    }.get(motion, [(0.0, 1.00, WIDTH*0.50, HEIGHT*0.50), (1.0, 1.22, WIDTH*0.50, HEIGHT*0.50)])

def interpolate_motion(keyframes, p):
    if p <= keyframes[0][0]:
        _, s, cx, cy = keyframes[0]; return s, cx, cy
    if p >= keyframes[-1][0]:
        _, s, cx, cy = keyframes[-1]; return s, cx, cy
    for i in range(len(keyframes) - 1):
        k0, k1 = keyframes[i], keyframes[i + 1]
        if k0[0] <= p <= k1[0]:
            span = k1[0] - k0[0]
            local = (p - k0[0]) / span if span > 0 else 1.0
            eased = ease_in_out(local)
            return (k0[1] + (k1[1] - k0[1]) * eased,
                    k0[2] + (k1[2] - k0[2]) * eased,
                    k0[3] + (k1[3] - k0[3]) * eased)
    _, s, cx, cy = keyframes[-1]; return s, cx, cy

# ============================================================
# 4 HÀM RENDER RIÊNG BIỆT
# ============================================================
def render_scene_kttv_v2(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_duration = max(1.5, min(duration - 0.8, duration * DRAW_DURATION_RATIO))
    draw_frames = int(draw_duration * FPS)
    retract_frames = int(0.35 * FPS)
    original_full = cv2.imread(str(image_path))
    if original_full is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    original_full = cv2.resize(original_full, (WIDTH, HEIGHT))
    title_band, content_bgr = split_title_band(original_full)
    white_content = np.full_like(content_bgr, 255)
    reveal_mask = np.zeros((CONTENT_H, WIDTH), dtype=np.uint8)
    zone_trajectories = extract_staggered_trajectories(image_path)
    all_pts_full = [p for z in zone_trajectories for p in z]
    if not all_pts_full:
        all_pts_full = [(WIDTH // 2, HEIGHT // 2)]
    all_points = trajectory_to_content_space(all_pts_full)
    phases = split_trajectory_into_phases(all_points)
    phase_frames = [int(draw_frames * PHASE_RATIOS[0]),
                    int(draw_frames * (PHASE_RATIOS[0] + PHASE_RATIOS[1])),
                    draw_frames]
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last_tip = all_points[0] if all_points else (WIDTH // 2, CONTENT_H // 2)
    motion_kfs = get_motion_keyframes(motion)

    for f_idx in range(total_frames):
        hand_visible = False
        hand_pos_x = hand_pos_y = 0
        if f_idx < draw_frames:
            if f_idx < phase_frames[0]:
                cp = 0; lp = f_idx / max(1, phase_frames[0])
            elif f_idx < phase_frames[1]:
                cp = 1; lp = (f_idx - phase_frames[0]) / max(1, phase_frames[1] - phase_frames[0])
            else:
                cp = 2; lp = (f_idx - phase_frames[1]) / max(1, phase_frames[2] - phase_frames[1])
            for pi in range(cp):
                for pt in phases[pi]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
            cpts = phases[cp]
            if cpts:
                cnt = max(1, min(int(lp * len(cpts)), len(cpts)))
                for pt in cpts[:cnt]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
                target = cpts[cnt - 1]
            else:
                target = last_tip
            hand_pos_x = target[0] + int(1.2 * math.sin(f_idx * 1.8))
            hand_pos_y = target[1] + int(1.2 * math.cos(f_idx * 1.8))
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (CONTENT_H + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255
        alpha = (cv2.GaussianBlur(reveal_mask, (13, 13), 0).astype(np.float32) / 255.0)[:, :, None]
        frame_content = (content_bgr * alpha + white_content * (1.0 - alpha)).astype(np.uint8)
        if hand_visible:
            paste_hand(frame_content, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)
        if f_idx < draw_frames + retract_frames:
            scale, cx_use, cy_use = 1.0, WIDTH * 0.5, CONTENT_H * 0.5
        else:
            op = (f_idx - draw_frames - retract_frames) / max(1, total_frames - draw_frames - retract_frames)
            st_, cxt, cyt = interpolate_motion(motion_kfs, op)
            cyc = (cyt / HEIGHT) * CONTENT_H
            blend = ease_in_out(min(1.0, op * 1.8))
            scale = 1.0 + (st_ - 1.0) * blend
            cx_use = WIDTH * 0.5 + (cxt - WIDTH * 0.5) * blend
            cy_use = CONTENT_H * 0.5 + (cyc - CONTENT_H * 0.5) * blend
        frame_out = compose_frame(title_band, crop_content_with_motion(frame_content, scale, cx_use, cy_use))
        proc.stdin.write(frame_out.tobytes())
    proc.stdin.close(); proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại (Chế độ 1)")

def render_scene_hybrid(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_duration = max(1.5, min(duration - 0.8, duration * DRAW_DURATION_RATIO))
    draw_frames = int(draw_duration * FPS)
    retract_frames = int(0.35 * FPS)
    original_full = cv2.imread(str(image_path))
    if original_full is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path}")
    original_full = cv2.resize(original_full, (WIDTH, HEIGHT))
    title_band, content_bgr = split_title_band(original_full)
    white_content = np.full_like(content_bgr, 255)
    reveal_mask = np.zeros((CONTENT_H, WIDTH), dtype=np.uint8)
    trajectory_full = extract_continuous_trajectory(image_path)
    trajectory = trajectory_to_content_space(trajectory_full)
    phases = split_trajectory_into_phases(trajectory)
    phase_frames = [int(draw_frames * PHASE_RATIOS[0]),
                    int(draw_frames * (PHASE_RATIOS[0] + PHASE_RATIOS[1])),
                    draw_frames]
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path, 320)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last_tip = trajectory[0] if trajectory else (WIDTH // 2, CONTENT_H // 2)
    smooth_cx, smooth_cy = float(last_tip[0]), float(last_tip[1])
    motion_kfs = get_motion_keyframes(motion)

    for f_idx in range(total_frames):
        hand_visible = False
        hand_pos_x = hand_pos_y = 0
        if f_idx < draw_frames:
            if f_idx < phase_frames[0]:
                cp = 0; lp = f_idx / max(1, phase_frames[0])
            elif f_idx < phase_frames[1]:
                cp = 1; lp = (f_idx - phase_frames[0]) / max(1, phase_frames[1] - phase_frames[0])
            else:
                cp = 2; lp = (f_idx - phase_frames[1]) / max(1, phase_frames[2] - phase_frames[1])
            for pi in range(cp):
                for pt in phases[pi]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
            cpts = phases[cp]
            if cpts:
                cnt = max(1, min(int(lp * len(cpts)), len(cpts)))
                for pt in cpts[:cnt]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
                target = cpts[cnt - 1]
            else:
                target = last_tip
            hand_pos_x = target[0] + int(1.2 * math.sin(f_idx * 1.8))
            hand_pos_y = target[1] + int(1.2 * math.cos(f_idx * 1.8))
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (CONTENT_H + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255
        alpha = (cv2.GaussianBlur(reveal_mask, (13, 13), 0).astype(np.float32) / 255.0)[:, :, None]
        frame_content = (content_bgr * alpha + white_content * (1.0 - alpha)).astype(np.uint8)
        if hand_visible:
            paste_hand(frame_content, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)
        if f_idx < draw_frames:
            scale = 1.0
            smooth_cx = smooth_cx * 0.95 + hand_pos_x * 0.05
            smooth_cy = smooth_cy * 0.95 + hand_pos_y * 0.05
            cx_use, cy_use = smooth_cx, smooth_cy
        elif f_idx < draw_frames + retract_frames:
            scale, cx_use, cy_use = 1.0, smooth_cx, smooth_cy
        else:
            op = (f_idx - draw_frames - retract_frames) / max(1, total_frames - draw_frames - retract_frames)
            st_, cxt, cyt = interpolate_motion(motion_kfs, op)
            cyc = (cyt / HEIGHT) * CONTENT_H
            blend = ease_in_out(min(1.0, op * 1.8))
            scale = 1.0 + (st_ - 1.0) * blend
            cx_use = smooth_cx + (cxt - smooth_cx) * blend
            cy_use = smooth_cy + (cyc - smooth_cy) * blend
        crop_w = int(WIDTH / scale); crop_h = int(CONTENT_H / scale)
        cx_clamped = max(crop_w // 2, min(WIDTH - crop_w // 2, int(cx_use)))
        cy_clamped = max(crop_h // 2, min(CONTENT_H - crop_h // 2, int(cy_use)))
        frame_out = compose_frame(title_band, crop_content_with_motion(frame_content, scale, cx_clamped, cy_clamped))
        proc.stdin.write(frame_out.tobytes())
    proc.stdin.close(); proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại (Chế độ 2)")

def render_scene_kttv_pure(image_path, duration, output_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    original_full = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    title_band, content_bgr = split_title_band(original_full)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    kfs = get_motion_keyframes(motion)
    for f_idx in range(total_frames):
        p = f_idx / max(1, total_frames - 1)
        s, cx, cyf = interpolate_motion(kfs, p)
        cy = (cyf / HEIGHT) * CONTENT_H
        frame_out = compose_frame(title_band, crop_content_with_motion(content_bgr, s, cx, cy))
        proc.stdin.write(frame_out.tobytes())
    proc.stdin.close(); proc.wait()

def render_scene_classic_hand(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_frames = int(max(1.5, min(duration - 0.8, duration * DRAW_DURATION_RATIO)) * FPS)
    retract_frames = int(0.35 * FPS)
    original_full = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    title_band, content_bgr = split_title_band(original_full)
    white_content = np.full_like(content_bgr, 255)
    reveal_mask = np.zeros((CONTENT_H, WIDTH), dtype=np.uint8)
    trajectory_full = extract_continuous_trajectory(image_path)
    trajectory = trajectory_to_content_space(trajectory_full)
    phases = split_trajectory_into_phases(trajectory)
    phase_frames = [int(draw_frames * PHASE_RATIOS[0]),
                    int(draw_frames * (PHASE_RATIOS[0] + PHASE_RATIOS[1])),
                    draw_frames]
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(output_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last_tip = trajectory[0] if trajectory else (WIDTH // 2, CONTENT_H // 2)
    for f_idx in range(total_frames):
        hand_visible = False
        if f_idx < draw_frames:
            if f_idx < phase_frames[0]:
                cp = 0; lp = f_idx / max(1, phase_frames[0])
            elif f_idx < phase_frames[1]:
                cp = 1; lp = (f_idx - phase_frames[0]) / max(1, phase_frames[1] - phase_frames[0])
            else:
                cp = 2; lp = (f_idx - phase_frames[1]) / max(1, phase_frames[2] - phase_frames[1])
            for pi in range(cp):
                for pt in phases[pi]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
            cpts = phases[cp]
            if cpts:
                cnt = max(1, min(int(lp * len(cpts)), len(cpts)))
                for pt in cpts[:cnt]:
                    cv2.circle(reveal_mask, pt, REVEAL_RADIUS, 255, -1)
                target = cpts[cnt - 1]
            else:
                target = last_tip
            hand_pos_x = target[0] + int(1.2 * math.sin(f_idx * 1.8))
            hand_pos_y = target[1] + int(1.2 * math.cos(f_idx * 1.8))
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (CONTENT_H + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255
        alpha = (cv2.GaussianBlur(reveal_mask, (13, 13), 0).astype(np.float32) / 255.0)[:, :, None]
        frame_content = (content_bgr * alpha + white_content * (1.0 - alpha)).astype(np.uint8)
        if hand_visible:
            paste_hand(frame_content, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)
        frame_out = compose_frame(title_band, frame_content)
        proc.stdin.write(frame_out.tobytes())
    proc.stdin.close(); proc.wait()

# ============================================================
# PLACEHOLDER IMAGE
# ============================================================
def create_placeholder_image(output_path, title):
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    f = font_for(48)
    text = "ẢNH KHÔNG TẠO ĐƯỢC"
    box = draw.textbbox((0, 0), text, font=f)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text(((WIDTH - tw) / 2, (HEIGHT - th) / 2), text, fill="#cccccc", font=f)
    img.save(output_path, quality=95)

# ============================================================
# PARALLEL GENERATION V5.4 — FAIR SHARE CAP + LIVE STATUS
# ============================================================
def parallel_generate_images(scenes, batch_dir, providers, image_timeout,
                              progress_state, flux_steps=4,
                              fair_share_enabled=True):
    """
    V5.4: 
    - Fair share cap: mỗi provider tối đa 1.5x share trung bình
    - Worker 1-lần-thử (không retry 2 tầng)
    - Không put lại queue khi fail
    - Fallback tuần tự có timeout, KHÔNG break (chuyển sang placeholder)
    """
    if not providers:
        raise RuntimeError("Không có provider nào.")

    total_scenes = len(scenes)
    if fair_share_enabled:
        fair_share_cap = max(3, int((total_scenes / len(providers)) * FAIR_SHARE_MULTIPLIER))
        st.caption(f"⚖️ Fair share cap: mỗi provider tối đa **{fair_share_cap}** cảnh "
                   f"({total_scenes} cảnh / {len(providers)} provider × {FAIR_SHARE_MULTIPLIER})")
    else:
        fair_share_cap = 999999  # Không giới hạn

    scene_queue = Queue()
    for i, s in enumerate(scenes):
        img_raw = batch_dir / f"scene_{i+1:03d}_raw.png"
        img = batch_dir / f"scene_{i+1:03d}.jpg"
        if not img.exists():
            scene_queue.put((i, s, img_raw, img))
            progress_state["scene_status"][i] = {
                "status": "pending", "provider": None, "started": None, "elapsed": 0.0
            }
        else:
            progress_state["scene_status"][i] = {
                "status": "done", "provider": "(cached)", "started": None, "elapsed": 0.0
            }

    total = scene_queue.qsize()
    if total == 0:
        return {}, 0

    results = {}
    failed_scenes = {}
    results_lock = threading.Lock()
    provider_stats = {p["name"]: {"ok": 0, "err": 0, "total_time": 0.0,
                                   "last_scene": None, "errors": []}
                      for p in providers}

    def worker(provider_cfg):
        name = provider_cfg["name"]
        my_count = 0
        while my_count < fair_share_cap:
            try:
                idx, scene, img_raw, img = scene_queue.get_nowait()
            except Empty:
                return

            t_start = time.time()
            with results_lock:
                progress_state["scene_status"][idx] = {
                    "status": "working", "provider": name,
                    "started": t_start, "elapsed": 0.0,
                }

            try:
                seed = hash(f"{scene['visual_prompt']}_{idx}") % (2**31)
                kwargs = provider_cfg.get("kwargs", {}).copy()
                if "seed" in provider_cfg.get("supports_seed", []):
                    kwargs["seed"] = seed
                data = provider_cfg["fn"](scene["visual_prompt"],
                                          *provider_cfg.get("args", []),
                                          timeout=image_timeout, **kwargs)
                if not data or len(data) < 500:
                    raise RuntimeError("empty data")
                save_image_from_bytes(data, img_raw)
                add_comic_overlays(img_raw, scene["title"],
                                   scene.get("callout_type", "speech"),
                                   scene.get("callout_text", ""),
                                   scene.get("callout_side", "right"), img)
                elapsed = time.time() - t_start
                my_count += 1
                with results_lock:
                    results[idx] = name
                    provider_stats[name]["ok"] += 1
                    provider_stats[name]["total_time"] += elapsed
                    provider_stats[name]["last_scene"] = idx + 1
                    progress_state["scene_status"][idx] = {
                        "status": "done", "provider": name,
                        "started": t_start, "elapsed": elapsed,
                    }
                    progress_state["done"] += 1
            except Exception as e:
                elapsed = time.time() - t_start
                err_msg = str(e)[:150]
                my_count += 1
                with results_lock:
                    failed_scenes[idx] = (scene, img_raw, img, f"{name}: {err_msg}")
                    provider_stats[name]["err"] += 1
                    provider_stats[name]["errors"].append(err_msg)
                    progress_state["scene_status"][idx] = {
                        "status": "failed", "provider": name,
                        "started": t_start, "elapsed": elapsed,
                        "error": err_msg,
                    }

    with ThreadPoolExecutor(max_workers=len(providers)) as ex:
        futures = [ex.submit(worker, p) for p in providers]
        wait(futures, timeout=None)

    # Fallback tuần tự — V5.4: KHÔNG break, tạo placeholder khi timeout
    if failed_scenes:
        start_fb = time.time()
        timed_out_count = 0
        st.warning(f"⚠️ {len(failed_scenes)} cảnh fail, fallback tuần tự (tối đa 5 phút)...")
        for idx in sorted(failed_scenes.keys()):
            scene, img_raw, img, _ = failed_scenes[idx]

            # Nếu đã vượt timeout → tạo placeholder NGAY, KHÔNG break
            if time.time() - start_fb > MAX_TOTAL_FALLBACK_TIME:
                create_placeholder_image(img, scene["title"])
                with results_lock:
                    results[idx] = "placeholder"
                    progress_state["done"] += 1
                    progress_state["scene_status"][idx] = {
                        "status": "placeholder", "provider": "placeholder",
                        "started": None, "elapsed": 0.0,
                    }
                timed_out_count += 1
                continue

            success = False
            for cfg in providers:
                try:
                    seed = hash(f"{scene['visual_prompt']}_{idx}") % (2**31)
                    kwargs = cfg.get("kwargs", {}).copy()
                    if "seed" in cfg.get("supports_seed", []):
                        kwargs["seed"] = seed
                    data = cfg["fn"](scene["visual_prompt"], *cfg.get("args", []),
                                     timeout=min(image_timeout, 45), **kwargs)
                    if data and len(data) > 500:
                        save_image_from_bytes(data, img_raw)
                        add_comic_overlays(img_raw, scene["title"],
                                           scene.get("callout_type", "speech"),
                                           scene.get("callout_text", ""),
                                           scene.get("callout_side", "right"), img)
                        with results_lock:
                            results[idx] = cfg["name"]
                            provider_stats[cfg["name"]]["ok"] += 1
                            progress_state["done"] += 1
                            progress_state["scene_status"][idx] = {
                                "status": "done", "provider": cfg["name"] + " (fallback)",
                                "started": None, "elapsed": 0.0,
                            }
                        success = True
                        break
                except Exception:
                    continue
            if not success:
                create_placeholder_image(img, scene["title"])
                with results_lock:
                    results[idx] = "placeholder"
                    progress_state["done"] += 1
                    progress_state["scene_status"][idx] = {
                        "status": "placeholder", "provider": "placeholder",
                        "started": None, "elapsed": 0.0,
                    }

        if timed_out_count > 0:
            st.warning(f"⚠️ {timed_out_count} cảnh timeout, đã dùng placeholder để video vẫn render được")

    progress_state["provider_stats"] = provider_stats
    return results, len(failed_scenes)

# ============================================================
# RENDER BATCH VỚI LIVE DASHBOARD
# ============================================================
def render_batch(batch_audio, scenes, batch_dir, hand_path, style,
                 cf_account, cf_token, hf_token, freetheai_key, together_key,
                 nexa_key, agnes_key, pollinations_key, pollinations_model,
                 image_timeout, flux_steps=4, fair_share_enabled=True):
    total = len(scenes)
    if total == 0:
        raise RuntimeError("Không có cảnh nào để render.")

    providers = build_provider_list(
        cf_account, cf_token, hf_token, freetheai_key, together_key,
        nexa_key, agnes_key, pollinations_key, pollinations_model, flux_steps)
    if not providers:
        raise RuntimeError("Chưa cấu hình provider ảnh nào.")

    st.markdown(f"### 🔗 {len(providers)} Provider tham gia")
    provider_cols = st.columns(min(4, len(providers)))
    for i, p in enumerate(providers):
        with provider_cols[i % len(provider_cols)]:
            st.markdown(f"**{i+1}.** {p['name']}")

    # ============================================
    # PHASE 1: Tạo ảnh song song — Live Dashboard
    # ============================================
    st.markdown("### 🎨 Tạo ảnh song song")

    progress_state = {
        "done": 0,
        "scene_status": {},
        "provider_stats": {p["name"]: {"ok": 0, "err": 0, "total_time": 0.0,
                                        "last_scene": None, "errors": []}
                            for p in providers},
    }
    progress_lock = threading.Lock()

    progress_bar = st.progress(0)
    progress_text = st.empty()
    stats_table = st.empty()
    scene_table = st.empty()

    def render_dashboard():
        with progress_lock:
            done = progress_state["done"]
            scene_status = dict(progress_state["scene_status"])
            pstats = dict(progress_state["provider_stats"])

        pct = min(1.0, done / total) * 0.6
        progress_bar.progress(pct)
        progress_text.markdown(f"**🎨 Tạo ảnh: {done}/{total}** ({pct/0.6*100:.0f}%)")

        stats_data = []
        for p in providers:
            name = p["name"]
            s = pstats.get(name, {"ok": 0, "err": 0, "total_time": 0.0,
                                   "last_scene": None, "errors": []})
            avg = s["total_time"] / s["ok"] if s["ok"] > 0 else 0
            last_err = s.get("errors", [])[-1][:40] if s.get("errors") else "—"
            stats_data.append({
                "Provider": name,
                "✅ OK": s["ok"],
                "❌ Lỗi": s["err"],
                "⏱️ TB (s)": f"{avg:.1f}" if avg > 0 else "—",
                "🎬 Cảnh cuối": f"#{s['last_scene']}" if s["last_scene"] else "—",
                "🐛 Lỗi gần nhất": last_err,
            })
        if stats_data:
            stats_table.dataframe(stats_data, use_container_width=True, hide_index=True)

        rows = []
        for i in range(total):
            st_info = scene_status.get(i, {"status": "pending", "provider": None, "elapsed": 0.0})
            status_icon = {
                "pending": "⏳ Chờ",
                "working": "🔄 Đang vẽ",
                "done": "✅ Xong",
                "failed": "❌ Lỗi",
                "placeholder": "⚠️ Placeholder",
            }.get(st_info["status"], "?")
            prov = st_info["provider"] or "—"
            elapsed = f"{st_info['elapsed']:.1f}s" if st_info.get("elapsed", 0) > 0 else "—"
            rows.append({
                "Cảnh": f"#{i+1:02d}",
                "Trạng thái": status_icon,
                "Provider": prov,
                "Thời gian": elapsed,
            })
        scene_table.dataframe(rows, use_container_width=True, hide_index=True,
                              height=min(400, 35 * total + 40))

    result_container = {"result": None, "error": None}
    def run_parallel():
        try:
            r, _ = parallel_generate_images(
                scenes, batch_dir, providers, image_timeout, progress_state,
                flux_steps, fair_share_enabled)
            result_container["result"] = r
        except Exception as e:
            result_container["error"] = e

    t = threading.Thread(target=run_parallel, daemon=True)
    t.start()
    while t.is_alive():
        render_dashboard()
        time.sleep(0.8)
    t.join()
    render_dashboard()

    if result_container["error"]:
        raise result_container["error"]

    used = result_container["result"] or {}
    stats = Counter(used.values())
    st.success(f"✅ Đã tạo {len(used)}/{total} ảnh — {dict(stats)}")

    # ============================================
    # PHASE 2: Render video
    # ============================================
    st.markdown("### 🎬 Render video")
    render_bar = st.progress(0)
    render_text = st.empty()

    scene_videos = []
    for i, s in enumerate(scenes, 1):
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"
        if not img.exists():
            raise RuntimeError(f"Thiếu ảnh scene {i}")
        duration = max(1.0, float(s["end"]) - float(s["start"]))
        motion = s.get("camera_motion", "zoom_in_center")

        if "1." in style or "Kiến Thức Thú Vị V2" in style:
            render_scene_kttv_v2(img, duration, vid, hand_path, motion)
        elif "2." in style or "Độc bản" in style or "Hybrid" in style:
            render_scene_hybrid(img, duration, vid, hand_path, motion)
        elif "3." in style or "Chỉ Camera" in style:
            render_scene_kttv_pure(img, duration, vid, motion)
        else:
            render_scene_classic_hand(img, duration, vid, hand_path, motion)

        scene_videos.append(vid)
        render_bar.progress(i / total)
        render_text.markdown(f"**🎬 Render: {i}/{total}** — {s['title']}")

    concat_file = batch_dir / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_videos), encoding="utf-8")
    batch_video = batch_dir / "batch_video.mp4"
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c", "copy", "-movflags", "+faststart", str(batch_video)], timeout=900)

    final_batch = batch_dir / "batch_final.mp4"
    run_cmd(["ffmpeg", "-y", "-i", str(batch_video), "-i", str(batch_audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
             "-b:a", "128k", "-movflags", "+faststart", str(final_batch)], timeout=900)

    for f in batch_dir.glob("scene_*_raw.png"):
        f.unlink(missing_ok=True)
    for f in batch_dir.glob("scene_*.mp4"):
        f.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    batch_video.unlink(missing_ok=True)
    return final_batch

def concat_batches(batch_videos, output_path):
    concat = output_path.parent / "batches.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in batch_videos), encoding="utf-8")
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
             "-c", "copy", "-movflags", "+faststart", str(output_path)], timeout=1800)

# ============================================================
# LUỒNG CHÍNH
# ============================================================
st.sidebar.divider()

if st.sidebar.button("🔎 KIỂM TRA PROVIDER", use_container_width=True):
    providers = build_provider_list(cf_account, cf_token, hf_token, freetheai_key,
                                     together_key, nexa_key, agnes_key,
                                     pollinations_key, pollinations_model, flux_steps)
    if not providers:
        st.error("Chưa có provider nào.")
    else:
        st.write(f"**{len(providers)} provider:**")
        for p in providers:
            st.write(f"• {p['name']}")
        if st.button("▶️ Test 1 ảnh"):
            tp = "2D comic doodle: a man standing at the edge of a cliff at sunset, red sunset, blue waves, white background, bold black outlines, no text"
            for cfg in providers:
                try:
                    with st.spinner(f"Test {cfg['name']}..."):
                        t0 = time.time()
                        data = cfg["fn"](tp, *cfg.get("args", []), timeout=45, **cfg.get("kwargs", {}))
                        elapsed = time.time() - t0
                    if data and len(data) > 500:
                        img = Image.open(io.BytesIO(data)).convert("RGB").resize((WIDTH, HEIGHT))
                        st.success(f"✅ {cfg['name']} — {elapsed:.1f}s")
                        st.image(img, use_container_width=True)
                        break
                except Exception as e:
                    st.warning(f"❌ {cfg['name']}: {str(e)[:150]}")
                    continue

audio = st.file_uploader("🎤 Tải lên voice", type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"])

if audio:
    st.audio(audio)
    if st.button("🚀 BẮT ĐẦU", type="primary", use_container_width=True):
        if not groq_key:
            st.error("Cần Groq API Key."); st.stop()

        root = Path(tempfile.mkdtemp(prefix="wb_v54_"))
        try:
            source = root / audio.name
            source.write_bytes(audio.getbuffer())
            duration = ffprobe_duration(source)
            st.info(f"Thời lượng: {duration/60:.2f} phút. Batch 10 phút.")

            client = groq_client(groq_key)
            batch_dir = root / "batches"; batch_dir.mkdir()
            chunks = chunk_audio(source, batch_dir)
            hand_path = Path("hand.png")

            batch_videos = []; all_scenes = 0
            status = st.empty()

            valid_chunks = [(bi, ch, ffprobe_duration(ch)) for bi, ch in enumerate(chunks)
                            if ffprobe_duration(ch) >= 5.0 or bi == 0]
            if not valid_chunks:
                st.error("Không có audio hợp lệ."); st.stop()

            camera_mode = ("random" if "Random" in camera_motion_mode else
                          f"fixed:{camera_motion_mode.replace('Cố định: ', '').strip()}"
                          if "Cố định" in camera_motion_mode else "auto")

            for idx, (bi, chunk, bdur) in enumerate(valid_chunks):
                bstart = bi * BATCH_SECONDS
                status.markdown(f"### 🧠 Đợt {idx+1}/{len(valid_chunks)} — Nhận diện giọng nói...")
                tr = transcribe_file(client, chunk, stt_model)
                segs = normalize_segments(tr, bstart)
                batch_text = "\n".join(f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}" for x in segs)

                status.markdown(f"### ✂️ Đợt {idx+1}/{len(valid_chunks)} — Lên kịch bản...")
                scenes = make_scene_plan(client, batch_text, bstart, bdur, planner_model,
                                         scene_min, scene_max, max_scenes, camera_mode)
                st.markdown(f"#### 📝 Đợt {idx+1}: {bdur:.1f}s → **{len(scenes)} cảnh**")
                with st.expander("Xem chi tiết các cảnh", expanded=False):
                    for si, s in enumerate(scenes, 1):
                        ci = f" | [{s.get('callout_type','').upper()}]: \"{s.get('callout_text','')}\"" if s.get('callout_text') else ""
                        st.caption(f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}{ci} | 🎥 {s.get('camera_motion','zoom_in_center')}")

                batch_work = root / f"work_{idx+1:03d}"; batch_work.mkdir()
                status.markdown(f"### 🎨 Đợt {idx+1}/{len(valid_chunks)} — Tạo ảnh + render...")
                bv = render_batch(chunk, scenes, batch_work, hand_path, draw_style,
                                  cf_account, cf_token, hf_token, freetheai_key, together_key,
                                  nexa_key, agnes_key, pollinations_key, pollinations_model,
                                  image_timeout, flux_steps, fair_share_enabled)
                saved = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bv, saved); batch_videos.append(saved)
                all_scenes += len(scenes)
                shutil.rmtree(batch_work, ignore_errors=True)

            status.markdown("### 🎬 Ghép video cuối...")
            final = root / "video_hoan_thien_final.mp4"
            concat_batches(batch_videos, final)
            st.success(f"Hoàn thành! {all_scenes} cảnh.")
            st.video(str(final))
            st.download_button("⬇️ TẢI VIDEO", data=final.read_bytes(),
                file_name="video_hoan_thien_final.mp4", mime="video/mp4",
                use_container_width=True)
        except Exception as e:
            st.exception(e)
