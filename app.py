"""
Xưởng Video Diễn Hoạt Kiến Thức AI — Bản Siêu Cấp V3
- Đa nhà cung cấp ảnh với fallback tự động
- Prompt sáng tạo đa dạng bố cục (không lặp khuôn bàn + bút)
- 8 kiểu camera motion cho từng cảnh
- Giữ nguyên 100% logic cốt lõi: hand animation, comic overlays, batch processing
"""

import os
import re
import io
import json
import math
import time
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
import cv2
import numpy as np

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
APP_TITLE = "Xưởng Video Diễn Hoạt Kiến Thức AI (Bản Siêu Cấp V3)"
BATCH_SECONDS = 5 * 60
FPS = 30
WIDTH = 1280
HEIGHT = 720

# --- Agnes AI ---
AGNES_API_URL = "https://apihub.agnes-ai.com/v1/images/generations"
AGNES_MODEL = "agnes-image-2.1-flash"

# --- Cloudflare Workers AI ---
CLOUDFLARE_BASE = "https://api.cloudflare.com/client/v4/accounts/"
CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# --- Hugging Face ---
HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_MODEL = "black-forest-labs/FLUX.1-schnell"

# --- FreeTheAi ---
FREETHEAI_BASE = "https://api.freetheai.xyz/v1/images/generations"

# --- Together AI ---
TOGETHER_BASE = "https://api.together.xyz/v1/images/generations"
TOGETHER_MODEL = "black-forest-labs/FLUX.1-schnell-Free"

# --- NexaAPI ---
NEXA_BASE = "https://api.nexa-api.com/v1/images/generations"
NEXA_MODEL = "flux-schnell"

# --- Pollinations (anonymous fallback) ---
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

# ============================================================
# GIAO DIỆN / CẤU HÌNH
# ============================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
st.title("🎬 Xưởng Video Diễn Hoạt Kiến Thức AI — Siêu Cấp V3")
st.caption("Groq (STT + Biên kịch) + 7 nhà cung cấp ảnh AI fallback + Sáng tạo bố cục đa dạng + 8 kiểu camera motion")

with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input(
        "Groq API Key",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password",
    )

    with st.expander("🎨 Nhà cung cấp ảnh AI", expanded=True):
        agnes_key = st.text_input(
            "Agnes AI API Key (Miễn phí)",
            value=st.secrets.get("AGNES_API_KEY", os.getenv("AGNES_API_KEY", "")),
            type="password",
            help="Lấy tại platform.agnes-ai.com",
        )
        cf_account = st.text_input(
            "Cloudflare Account ID",
            value=st.secrets.get("CLOUDFLARE_ACCOUNT_ID", os.getenv("CLOUDFLARE_ACCOUNT_ID", "")),
            type="password",
        )
        cf_token = st.text_input(
            "Cloudflare API Token",
            value=st.secrets.get("CLOUDFLARE_API_TOKEN", os.getenv("CLOUDFLARE_API_TOKEN", "")),
            type="password",
        )
        hf_token = st.text_input(
            "Hugging Face Token (Miễn phí)",
            value=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "")),
            type="password",
        )
        freetheai_key = st.text_input(
            "FreeTheAi API Key (Miễn phí)",
            value=st.secrets.get("FREETHEAI_API_KEY", os.getenv("FREETHEAI_API_KEY", "")),
            type="password",
        )
        together_key = st.text_input(
            "Together AI API Key",
            value=st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY", "")),
            type="password",
        )
        nexa_key = st.text_input(
            "NexaAPI Key ($5 free)",
            value=st.secrets.get("NEXA_API_KEY", os.getenv("NEXA_API_KEY", "")),
            type="password",
        )

    st.header("🧠 Mô hình Groq")
    stt_model = st.selectbox(
        "STT Model",
        ["whisper-large-v3", "whisper-large-v3-turbo"],
        index=0,
    )
    planner_model = st.selectbox(
        "Biên kịch Model",
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0,
    )

    st.header("🎬 Phong cách diễn hoạt")
    draw_style = st.selectbox(
        "Chọn phong cách",
        [
            "1. Kiến Thức Thú Vị V2 (Vẽ tuần tự + Bong bóng thoại + Pan/Zoom)",
            "2. Độc bản Hybrid (Tay vẽ + Camera Steadicam)",
            "3. Chỉ Camera Pan & Zoom (ẩn bàn tay)",
            "4. Bảng trắng cổ điển (Tay vẽ góc máy tĩnh)",
        ],
        index=0,
    )

    st.header("🎥 Camera Motion")
    camera_motion_mode = st.selectbox(
        "Chế độ chuyển động camera",
        [
            "Auto (AI chọn cho từng cảnh)",
            "Random (Code chọn ngẫu nhiên)",
            "Cố định: zoom_in_center",
            "Cố định: zoom_out_center",
            "Cố định: pan_left_to_right",
            "Cố định: pan_right_to_left",
            "Cố định: ken_burns_slow",
        ],
        index=0,
    )

    st.header("⏱️ Khóa nhịp cảnh")
    scene_min = st.slider("Tối thiểu (giây)", 18, 25, 19)
    scene_max = st.slider("Tối đa (giây)", 22, 35, 27)
    if scene_max < scene_min:
        scene_max = scene_min

    st.header("⚙️ Cài đặt khác")
    max_scenes = st.slider("Số cảnh tối đa mỗi batch", 5, 20, 14)
    image_timeout = st.slider("Timeout tạo ảnh (giây)", 30, 180, 120)
    flux_steps = st.slider("Số bước FLUX (cao = nét hơn, chậm hơn)", 4, 8, 4)

# ============================================================
# TIỆN ÍCH HỆ THỐNG
# ============================================================
def run_cmd(cmd, timeout=600):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-5000:] or "Lệnh hệ thống thất bại")
    return p.stdout

def ffprobe_duration(path):
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], timeout=60)
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
    raise ValueError("AI không phản hồi cấu trúc JSON hợp lệ")

def groq_client(key):
    return Groq(api_key=key)

def transcribe_file(client, path, model):
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(Path(path).name, f.read()),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="vi",
            temperature=0.0,
        )
    return result

def chunk_audio(src, out_dir):
    pattern = str(Path(out_dir) / "batch_%03d.m4a")
    run_cmd([
        "ffmpeg", "-y", "-i", str(src),
        "-map", "0:a:0",
        "-c:a", "aac", "-b:a", "96k",
        "-f", "segment", "-segment_time", str(BATCH_SECONDS),
        "-reset_timestamps", "1",
        pattern,
    ], timeout=900)
    return sorted(Path(out_dir).glob("batch_*.m4a"))

def normalize_segments(result, offset):
    data = result.model_dump() if hasattr(result, "model_dump") else result
    segments = data.get("segments", []) if isinstance(data, dict) else getattr(result, "segments", [])
    out = []
    for s in segments or []:
        if isinstance(s, dict):
            stt = float(s.get("start", 0))
            end = float(s.get("end", stt))
            text = str(s.get("text", "")).strip()
        else:
            stt = float(getattr(s, "start", 0))
            end = float(getattr(s, "end", stt))
            text = str(getattr(s, "text", "")).strip()
        if text:
            out.append({"start": stt + offset, "end": end + offset, "text": text})
    return out

def sanitize_prompt_text(prompt):
    replacements = {
        r"\bblood\b": "dark ink",
        r"\bbleed\b": "drip",
        r"\bsuicide\b": "despair",
        r"\bkill(ing|er)?\b": "oppression",
        r"\bdead\b": "fallen",
        r"\bdeath\b": "crisis",
        r"\bcorpse\b": "shadow",
        r"\bjump(ing)?\b": "falling shadow",
        r"\bweapon\b": "heavy chain",
    }
    cleaned = prompt
    for pattern, rep in replacements.items():
        cleaned = re.sub(pattern, rep, cleaned, flags=re.IGNORECASE)
    return cleaned

# ============================================================
# BỘ ĐIỀU PHỐI KỊCH BẢN (NÂNG CẤP: SÁNG TẠO BỐ CỤC + CAMERA MOTION)
# ============================================================
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model,
                    min_s, max_s, max_scenes, camera_mode="auto"):
    expected_scenes = max(1, round(batch_duration / 23.0))

    system = f"""
Bạn là giám đốc sáng tạo kịch bản cho kênh hoạt họa kiến thức phong cách "Kiến Thức Thú Vị".
Nhiệm vụ: Chia đoạn âm thanh {batch_duration:.0f}s thành khoảng {expected_scenes} cảnh lớn, mạch lạc ({min_s}-{max_s}s/cảnh).

QUY TẮC VỀ TIÊU ĐỀ ("title"):
- Tiếng Việt tự nhiên, 3-6 từ, VIẾT HOA, tóm tắt luận điểm chính.
- KHÔNG dùng từ ghép kiểu dịch máy (như 'đán mọc', 'đáng người', 'vô vì').
- Ví dụ tốt: "CỐ GẮNG HÀI LÒNG MỌI NGƯỜI", "NỖI SỢ BỊ PHÁN XÉT", "ĐÁNH MẤT BẢN THÂN".

QUY TẮC VỀ CHỮ TRÊN TRANH ("callout_type", "callout_text"):
- "speech": bong bóng thoại (nhân vật nói: "KHỔ QUÁ RỒI!", "LẠI PHẢI NHẬN À?")
- "thought": đám mây suy nghĩ (nhân vật tự vấn: "HỌ CÓ GHÉT MÌNH KHÔNG?")
- "sticker": nhãn dán nhấn mạnh: "BẪY TÂM LÝ!", "MẤT HẾT TỰ DO!"
- "none": không có chữ
- LUÂN PHIÊN thay đổi, đừng cảnh nào cũng speech.

QUY TẮC QUAN TRỌNG NHẤT — MÔ TẢ TRANH ("visual_prompt") PHẢI SÁNG TẠO VÀ ĐA DẠNG:

MỖI CẢNH LÀ MỘT "SÂN KHẤU" KHÁC NHAU. TUYỆT ĐỐI KHÔNG lặp lại bố cục.
KHÔNG được bắt đầu nhiều cảnh bằng cùng một mô tả. KHÔNG có khuôn mẫu "nhân vật ngồi bàn có bút".

Mô tả trong visual_prompt (tiếng Anh), phải bao gồm 4-5 yếu tố:
1. NHÂN VẬT + TƯ THẾ đa dạng: đứng, ngồi xổm, chạy, ngã, chỉ tay, ôm đầu, đứng giữa biển, đứng trên núi, bay lơ lửng, chui vào hộp, bị trói, đang bơi, đang leo, đang chiến đấu...
2. HÀNH ĐỘNG cụ thể: ký giấy, đóng dấu, bị kéo, bị đẩy, gánh nặng, đang rơi, đang đuổi theo, đang trốn...
3. BỐI CẢNH khác nhau: bãi biển hoàng hôn, thành phố, sa mạc, mê cung, đấu trường, văn phòng, trên mây, dưới nước, trong bóng tối, trên đỉnh núi, trong rừng...
4. ĐỒ VẬT ẨN DỤ: dây xích, đồng hồ cát, con rối, tảng đá, quả bóng, cánh cửa, gương soi, la bàn, chìa khóa, mê cung...
5. CẢM XÚC rõ ràng: buồn, sợ, giận, ngạc nhiên, kiệt sức, cô đơn, tự tin, do dự...
6. MÀU NHẤN: 1-2 màu (đỏ/xanh/cam/vàng) cho điểm quan trọng.

VÍ DỤ TỐT (đa dạng bố cục):
- "2D comic doodle: a young man standing at the edge of a cliff at sunset, looking down at a vast ocean of papers below, red sunset, blue waves, white background, bold black outlines, no text"
- "2D comic doodle: a man trapped inside a giant glass jar, hands pressing against the walls, other people watching from outside pointing, red accents on the jar edges, white background, no text"
- "2D comic doodle: a man kneeling on the ground, carrying a mountain of heavy rocks on his back labeled with symbols, sweat drops, red spot color on the heaviest rock, white background, no text"
- "2D comic doodle: a man running on a treadmill that is actually a giant clock, chains held by hands reaching from off-screen, red clock hands, white background, no text"
- "2D comic doodle: a man standing on top of a giant question mark, surrounded by floating question mark bubbles, confused expression, blue and red accents, white background, no text"
- "2D comic doodle: a man in a small boat on a stormy sea, holding an umbrella made of paper contracts, lightning above, blue waves and red lightning, white background, no text"
- "2D comic doodle: a man in the center of a labyrinth of mirrors, his reflection shown in multiple distorted versions, red reflection accents, white background, no text"

VÍ DỤ XẤU (bị cấm — vì lặp khuôn):
- "a man sitting at a desk with papers on the left" ← lặp bố cục
- "a waist-up man with contract papers and red arrow" ← khuôn cũ

CÁC RÀNG BUỘC PHONG CÁCH (CHỈ về style, KHÔNG áp bố cục):
- Phong cách: 2D comic doodle, nét mực đen dày, nét vẽ tay ngộ nghĩnh.
- Nền: TRẮNG TINH hoặc có yếu tố bối cảnh đơn giản (biển, mây, sa mạc...).
- KHÔNG chữ, KHÔNG số, KHÔNG bong bóng thoại rỗng (tool sẽ tự vẽ chữ overlay sau).
- Có thể có bàn/đồ vật nếu phù hợp nội dung, KHÔNG bắt buộc.

QUY TẮC CAMERA MOTION ("camera_motion"):
Với mỗi cảnh, chọn MỘT kiểu chuyển động phù hợp với cảm xúc/nội dung:
- "zoom_in_center": phóng to vào trung tâm (nhấn mạnh, gay cấn)
- "zoom_out_center": thu nhỏ ra (giải tỏa, kết thúc)
- "pan_left_to_right": lướt ngang trái→phải (giới thiệu, dẫn dắt)
- "pan_right_to_left": lướt ngang phải→trái (hồi tưởng, quay lại)
- "zoom_in_top_left": phóng to góc trên trái (tò mò, khám phá)
- "zoom_in_bottom_right": phóng to góc dưới phải (chú ý chi tiết)
- "ken_burns_slow": chuyển động chậm kết hợp pan + zoom (mơ màng)
- "static": đứng yên (nhấn mạnh nội dung tĩnh)
LUÂN PHIÊN, tránh lặp lại liên tiếp cùng một kiểu.

LUÔN LUÔN trả về đúng JSON.

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0, "end": 22.0,
      "title": "NỖI SỢ BỊ PHÁN XÉT",
      "callout_type": "thought", "callout_text": "TỚ ĐANG NGHĨ GÌ?", "callout_side": "right",
      "camera_motion": "zoom_in_center",
      "visual_prompt": "2D comic doodle: a young man standing alone on a small floating island in the middle of a vast white void, giant floating eyes watching him from all directions, sweat drops, red accents on the eyes, pure white background, bold black outlines, no text"
    }}
  ]
}}
"""
    user = f"""
Độ dài âm thanh: {batch_duration:.2f} giây.
CHỈ ĐƯỢC CHIA TỐI ĐA {expected_scenes} CẢNH (Mỗi cảnh dài {min_s}-{max_s} giây).

TRANSCRIPT:
{transcript_text}
"""
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.15,
            max_tokens=12000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        obj = extract_json(r.choices[0].message.content)
        raw_scenes = obj.get("scenes", [])
    except Exception:
        raw_scenes = []

    valid_motions = {
        "zoom_in_center", "zoom_out_center", "pan_left_to_right", "pan_right_to_left",
        "zoom_in_top_left", "zoom_in_bottom_right", "ken_burns_slow", "static",
    }

    clean = []
    for s in raw_scenes[:max_scenes]:
        try:
            a = max(0.0, float(s["start"]))
            b = min(batch_duration, float(s["end"]))
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
            clean.append({
                "start": a,
                "end": b,
                "title": str(s.get("title", "BÀI HỌC KIẾN THỨC")).strip().upper(),
                "callout_type": ct,
                "callout_text": str(s.get("callout_text", "")).strip(),
                "callout_side": str(s.get("callout_side", "right")).strip().lower(),
                "camera_motion": cm,
                "visual_prompt": vp,
            })
        except Exception:
            continue

    if not clean:
        clean = [{
            "start": 0.0,
            "end": batch_duration,
            "title": "BÀI HỌC QUAN TRỌNG",
            "callout_type": "speech",
            "callout_text": "RẤT SAI LẦM!",
            "callout_side": "right",
            "camera_motion": "zoom_in_center",
            "visual_prompt": "2D comic doodle: a man standing at the edge of a cliff at sunset, looking down at a vast ocean of papers below, red sunset, blue waves, white background, bold black outlines, no text",
        }]

    # Gộp cảnh ngắn tự động (< 17s)
    merged = []
    for s in clean:
        if not merged:
            merged.append(s)
        else:
            prev = merged[-1]
            dur = s["end"] - s["start"]
            if dur < 17.0 or (s["start"] - prev["start"] < 17.0):
                prev["end"] = max(prev["end"], s["end"])
                if not prev.get("callout_text") and s.get("callout_text"):
                    prev["callout_text"] = s["callout_text"]
                    prev["callout_type"] = s["callout_type"]
            else:
                merged.append(s)

    clean = merged
    clean[0]["start"] = 0.0
    for i in range(len(clean) - 1):
        clean[i]["end"] = clean[i + 1]["start"]
    clean[-1]["end"] = batch_duration

    # Chia cảnh dài > 35s
    final_scenes = []
    for s in clean:
        dur = s["end"] - s["start"]
        if dur > 35.0:
            mid = s["start"] + dur / 2.0
            final_scenes.append({
                "start": s["start"],
                "end": mid,
                "title": s["title"],
                "callout_type": s.get("callout_type", "speech"),
                "callout_text": s.get("callout_text", ""),
                "callout_side": s.get("callout_side", "right"),
                "camera_motion": s.get("camera_motion", "zoom_in_center"),
                "visual_prompt": s["visual_prompt"],
            })
            final_scenes.append({
                "start": mid,
                "end": s["end"],
                "title": f"{s['title']} (TIẾP)",
                "callout_type": "sticker",
                "callout_text": "CẦN CẨN TRỌNG!",
                "callout_side": "right",
                "camera_motion": "zoom_out_center",
                "visual_prompt": s["visual_prompt"] + ", continuation scene, different angle, clean white background",
            })
        else:
            final_scenes.append(s)

    # Nếu mode là "random", override camera_motion bằng random
    if camera_mode == "random":
        import random
        motions_list = list(valid_motions - {"static"})
        for s in final_scenes:
            s["camera_motion"] = random.choice(motions_list)
    elif camera_mode.startswith("fixed:"):
        fixed_motion = camera_mode.split(":", 1)[1].strip()
        if fixed_motion in valid_motions:
            for s in final_scenes:
                s["camera_motion"] = fixed_motion

    return final_scenes

# ============================================================
# MULTI-PROVIDER IMAGE ENGINES
# ============================================================
def _build_full_prompt(prompt):
    """Chỉ giữ style constraints, KHÔNG áp bố cục. Để AI tự do sáng tạo theo visual_prompt."""
    safe = sanitize_prompt_text(prompt)
    return f"""{safe}.

STYLE CONSTRAINTS (chỉ về phong cách vẽ, KHÔNG áp bố cục):
- Authentic 2D comic doodle art style, thick black ink contour outlines, hand-drawn wobbly lines.
- Pure solid flat white background OR simple scene background (ocean, clouds, desert, void) if described above.
- Vivid expressive cartoon character with clear emotion matching the description.
- Selective vibrant spot colors (red, blue, orange, green) ONLY on key symbolic elements.
- Absolutely NO text, NO letters, NO numbers, NO captions, NO empty speech balloons anywhere in the image.
- Do not draw any desk, table, markers, pens, or erasers UNLESS the description explicitly mentions them.
- Wide 16:9 cinematic composition, subject centered or off-center based on drama, leaving negative space.
"""

def _validate_image_bytes(data, provider_name):
    """Kiểm tra dữ liệu trả về có phải ảnh hợp lệ không."""
    if not data or len(data) < 500:
        raise RuntimeError(f"{provider_name}: dữ liệu quá nhỏ ({len(data) if data else 0} bytes)")
    if not (data[:3] == b'\xff\xd8\xff' or data[:8] == b'\x89PNG\r\n\x1a\n' or data[:4] == b'RIFF'):
        preview = data[:150].decode("utf-8", errors="ignore").lower()
        if "<html" in preview or "<!doctype" in preview:
            raise RuntimeError(f"{provider_name}: trả về HTML (server quá tải hoặc bị chặn)")
        raise RuntimeError(f"{provider_name}: dữ liệu không phải ảnh (thiếu magic bytes)")
    return data

# --- 1. AGNES AI ---
def agnes_image_request(prompt, api_key, timeout=120):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Agnes AI: chưa có API key")
    full_prompt = _build_full_prompt(prompt)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": AGNES_MODEL,
        "prompt": full_prompt,
        "size": "1280x720",
        "extra_body": {"response_format": "b64_json"},
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(AGNES_API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("Agnes AI: rate limit sau 3 lần thử")
            if r.status_code >= 400:
                raise RuntimeError(f"Agnes AI HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            item = data.get("data", [{}])[0]
            if item.get("b64_json"):
                return _validate_image_bytes(base64.b64decode(item["b64_json"]), "Agnes AI")
            if item.get("url"):
                img = requests.get(item["url"], timeout=timeout)
                return _validate_image_bytes(img.content, "Agnes AI (URL)")
            raise RuntimeError(f"Agnes AI: response không có ảnh: {str(data)[:300]}")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Agnes AI timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Agnes AI lỗi: {e}")

# --- 2. CLOUDFLARE WORKERS AI ---
def cloudflare_image_request(prompt, account_id, api_token, timeout=120, steps=4):
    account_id = (account_id or "").strip()
    api_token = (api_token or "").strip()
    if not account_id or not api_token:
        raise RuntimeError("Cloudflare: chưa có Account ID hoặc Token")
    url = f"{CLOUDFLARE_BASE}{account_id}/ai/run/{CLOUDFLARE_MODEL}"
    full_prompt = _build_full_prompt(prompt)
    payload = {"prompt": full_prompt, "steps": steps}
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(3 * attempt)
                    continue
                raise RuntimeError("Cloudflare: hết quota 10k neurons/ngày")
            if r.status_code >= 400:
                raise RuntimeError(f"Cloudflare HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            if not data.get("success", True):
                raise RuntimeError(f"Cloudflare lỗi: {str(data)[:300]}")
            b64 = data.get("result", {}).get("image")
            if not b64:
                raise RuntimeError("Cloudflare: không có trường image trong response")
            return _validate_image_bytes(base64.b64decode(b64), "Cloudflare")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Cloudflare timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Cloudflare lỗi: {e}")

# --- 3. HUGGING FACE ---
def hf_image_request(prompt, token, timeout=120):
    token = (token or "").strip()
    if not token:
        raise RuntimeError("Hugging Face: chưa có token")
    url = f"{HF_API_URL}{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    full_prompt = _build_full_prompt(prompt)
    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers=headers, json={"inputs": full_prompt}, timeout=timeout)
            if r.status_code == 503:
                if attempt < 3:
                    time.sleep(10 * attempt)
                    continue
                raise RuntimeError("Hugging Face: model đang load, thử lại sau")
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("Hugging Face: hết quota miễn phí")
            if r.status_code >= 400:
                raise RuntimeError(f"Hugging Face HTTP {r.status_code}: {r.text[:300]}")
            return _validate_image_bytes(r.content, "Hugging Face")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Hugging Face timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Hugging Face lỗi: {e}")

# --- 4. FREETHEAI ---
def freetheai_image_request(prompt, api_key, timeout=120):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("FreeTheAi: chưa có API key")
    full_prompt = _build_full_prompt(prompt)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "flux", "prompt": full_prompt, "n": 1, "size": "1280x720"}
    for attempt in range(1, 4):
        try:
            r = requests.post(FREETHEAI_BASE, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("FreeTheAi: rate limit")
            if r.status_code >= 400:
                raise RuntimeError(f"FreeTheAi HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            item = data.get("data", [{}])[0]
            if item.get("b64_json"):
                return _validate_image_bytes(base64.b64decode(item["b64_json"]), "FreeTheAi")
            if item.get("url"):
                img = requests.get(item["url"], timeout=timeout)
                return _validate_image_bytes(img.content, "FreeTheAi (URL)")
            raise RuntimeError(f"FreeTheAi: response không có ảnh: {str(data)[:300]}")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"FreeTheAi timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"FreeTheAi lỗi: {e}")

# --- 5. TOGETHER AI ---
def together_image_request(prompt, api_key, timeout=120):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("Together AI: chưa có API key")
    full_prompt = _build_full_prompt(prompt)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": TOGETHER_MODEL,
        "prompt": full_prompt,
        "width": WIDTH,
        "height": HEIGHT,
        "steps": 4,
        "n": 1,
        "response_format": "b64_json",
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(TOGETHER_BASE, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("Together AI: rate limit")
            if r.status_code >= 400:
                raise RuntimeError(f"Together AI HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            item = data.get("data", [{}])[0]
            if item.get("b64_json"):
                return _validate_image_bytes(base64.b64decode(item["b64_json"]), "Together AI")
            if item.get("url"):
                img = requests.get(item["url"], timeout=timeout)
                return _validate_image_bytes(img.content, "Together AI (URL)")
            raise RuntimeError(f"Together AI: response không có ảnh: {str(data)[:300]}")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Together AI timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Together AI lỗi: {e}")

# --- 6. NEXAAPI ---
def nexa_image_request(prompt, api_key, timeout=120):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("NexaAPI: chưa có API key")
    full_prompt = _build_full_prompt(prompt)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": NEXA_MODEL, "prompt": full_prompt, "width": WIDTH, "height": HEIGHT, "n": 1}
    for attempt in range(1, 4):
        try:
            r = requests.post(NEXA_BASE, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError("NexaAPI: rate limit")
            if r.status_code >= 400:
                raise RuntimeError(f"NexaAPI HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            item = data.get("data", [{}])[0]
            if item.get("b64_json"):
                return _validate_image_bytes(base64.b64decode(item["b64_json"]), "NexaAPI")
            if item.get("url"):
                img = requests.get(item["url"], timeout=timeout)
                return _validate_image_bytes(img.content, "NexaAPI (URL)")
            raise RuntimeError(f"NexaAPI: response không có ảnh: {str(data)[:300]}")
        except requests.exceptions.Timeout:
            if attempt < 3:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"NexaAPI timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"NexaAPI lỗi: {e}")

# --- 7. POLLINATIONS (anonymous fallback) ---
def pollinations_image_request(prompt, timeout=120, seed=None):
    full_prompt = _build_full_prompt(prompt)
    encoded = requests.utils.quote(full_prompt, safe="")
    url = f"{POLLINATIONS_BASE}{encoded}"
    params = {"width": WIDTH, "height": HEIGHT, "model": "flux", "nologo": "true"}
    if seed is not None:
        params["seed"] = seed
    for attempt in range(1, 3):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"Accept": "image/*"})
            if r.status_code == 429:
                if attempt < 2:
                    time.sleep(15)
                    continue
                raise RuntimeError("Pollinations: rate limit 15s")
            if r.status_code >= 400:
                raise RuntimeError(f"Pollinations HTTP {r.status_code}: {r.text[:200]}")
            return _validate_image_bytes(r.content, "Pollinations")
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
            else:
                raise RuntimeError(f"Pollinations timeout sau {timeout}s")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                raise RuntimeError(f"Pollinations lỗi: {e}")

# ============================================================
# FALLBACK ORCHESTRATOR
# ============================================================
def generate_image_with_fallback(prompt, provider_configs, timeout=120, seed=None):
    """Thử từng nhà cung cấp theo thứ tự ưu tiên. Trả về (image_bytes, provider_name)."""
    errors = []
    for cfg in provider_configs:
        name = cfg["name"]
        try:
            kwargs = cfg.get("kwargs", {}).copy()
            if "seed" in cfg.get("supports_seed", []):
                kwargs["seed"] = seed
            data = cfg["fn"](prompt, *cfg.get("args", []), timeout=timeout, **kwargs)
            if data and len(data) > 500:
                return data, name
        except Exception as e:
            err_msg = str(e)[:200]
            errors.append(f"{name}: {err_msg}")
            continue

    error_summary = "\n".join(f"• {e}" for e in errors) if errors else "Không có nhà cung cấp nào được cấu hình"
    raise RuntimeError(f"Tất cả nhà cung cấp đều thất bại:\n{error_summary}")

def build_provider_list(cf_account, cf_token, hf_token, freetheai_key,
                         together_key, nexa_key, agnes_key, flux_steps=4):
    """Xây dựng danh sách nhà cung cấp theo thứ tự ưu tiên, chỉ gồm những cái có key."""
    providers = []
    if agnes_key and agnes_key.strip():
        providers.append({"name": "Agnes AI", "fn": agnes_image_request, "args": [agnes_key.strip()]})
    if cf_account and cf_token and cf_account.strip() and cf_token.strip():
        providers.append({
            "name": "Cloudflare",
            "fn": cloudflare_image_request,
            "args": [cf_account.strip(), cf_token.strip()],
            "kwargs": {"steps": flux_steps},
        })
    if hf_token and hf_token.strip():
        providers.append({"name": "Hugging Face", "fn": hf_image_request, "args": [hf_token.strip()]})
    if freetheai_key and freetheai_key.strip():
        providers.append({"name": "FreeTheAi", "fn": freetheai_image_request, "args": [freetheai_key.strip()]})
    if together_key and together_key.strip():
        providers.append({"name": "Together AI", "fn": together_image_request, "args": [together_key.strip()]})
    if nexa_key and nexa_key.strip():
        providers.append({"name": "NexaAPI", "fn": nexa_image_request, "args": [nexa_key.strip()]})
    # Pollinations luôn có sẵn làm fallback cuối cùng
    providers.append({
        "name": "Pollinations (anonymous)",
        "fn": pollinations_image_request,
        "args": [],
        "kwargs": {"seed": None},
        "supports_seed": ["seed"],
    })
    return providers

def save_image_from_bytes(data, output_path):
    """Lưu bytes ảnh vào file, resize về chuẩn video."""
    if not data:
        raise RuntimeError("Dữ liệu ảnh rỗng (None/empty)")
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
# FONT & OVERLAY COMIC
# ============================================================
def font_for(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def add_comic_overlays(image_path, title, callout_type, callout_text, callout_side, output_path):
    img = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)

    if title:
        band_h = 95
        draw.rectangle([0, 0, WIDTH, band_h], fill="white")
        SAFE_TITLE_WIDTH = 850
        f_size = 36
        f_title = font_for(f_size)
        while f_size > 18:
            box = draw.textbbox((0, 0), title, font=f_title)
            if box[2] - box[0] <= SAFE_TITLE_WIDTH:
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
        if callout_side == "left":
            cx, cy = int(WIDTH * 0.28), int(HEIGHT * 0.35)
        else:
            cx, cy = int(WIDTH * 0.74), int(HEIGHT * 0.32)

        if callout_type == "speech":
            pad_x, pad_y = 18, 12
            rect = [cx - bw // 2 - pad_x, cy - bh // 2 - pad_y, cx + bw // 2 + pad_x, cy + bh // 2 + pad_y]
            draw.rounded_rectangle(rect, radius=14, fill="white", outline="black", width=4)
            tail_tip = (cx - 20, cy + bh // 2 + pad_y + 18)
            tail_poly = [(cx - 32, cy + bh // 2 + pad_y - 2), (cx - 8, cy + bh // 2 + pad_y - 2), tail_tip]
            draw.polygon(tail_poly, fill="white", outline="black")
            draw.line([(cx - 30, cy + bh // 2 + pad_y), (cx - 10, cy + bh // 2 + pad_y)], fill="white", width=5)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill="#1b5e20", font=f_text)
        elif callout_type == "thought":
            pad_x, pad_y = 22, 14
            rect = [cx - bw // 2 - pad_x, cy - bh // 2 - pad_y, cx + bw // 2 + pad_x, cy + bh // 2 + pad_y]
            draw.rounded_rectangle(rect, radius=24, fill="white", outline="black", width=3)
            draw.ellipse([cx - 20, cy + bh // 2 + pad_y + 4, cx - 10, cy + bh // 2 + pad_y + 14], fill="white", outline="black", width=3)
            draw.ellipse([cx - 28, cy + bh // 2 + pad_y + 17, cx - 22, cy + bh // 2 + pad_y + 23], fill="white", outline="black", width=2)
            draw.ellipse([cx - 34, cy + bh // 2 + pad_y + 26, cx - 30, cy + bh // 2 + pad_y + 30], fill="white", outline="black", width=2)
            draw.text((cx - bw // 2, cy - bh // 2 - 2), callout_text, fill="#0d47a1", font=f_text)
        else:
            pad_x, pad_y = 16, 9
            bx, by = int(WIDTH * 0.75), int(HEIGHT * 0.88)
            b_rect = [bx - bw // 2 - pad_x, by - bh // 2 - pad_y, bx + bw // 2 + pad_x, by + bh // 2 + pad_y]
            draw.rounded_rectangle(b_rect, radius=8, fill="white", outline="#b71c1c", width=4)
            draw.text((bx - bw // 2, by - bh // 2 - 2), callout_text, fill="#b71c1c", font=f_text)

    img.save(output_path, quality=95)

# ============================================================
# BÀN TAY & TRIỆT TIÊU BÓNG MỜ (GIỮ NGUYÊN)
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
    for candidate in [hand_path, Path("hand.png"), Path("assets/hand.png")]:
        if candidate and Path(candidate).exists():
            p = Path(candidate)
            break
    if p:
        try:
            pil_hand = Image.open(p).convert("RGBA")
        except Exception:
            pil_hand = generate_fallback_hand()
    else:
        pil_hand = generate_fallback_hand()

    w, h = pil_hand.size
    new_h = int(h * (target_width / w))
    pil_hand = pil_hand.resize((target_width, new_h), Image.Resampling.LANCZOS)
    hand_np = np.array(pil_hand)

    bgr = cv2.cvtColor(hand_np[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = hand_np[:, :, 3]

    # Khử bóng mờ hình chữ nhật
    alpha[:6, :] = 0
    alpha[-6:, :] = 0
    alpha[:, :6] = 0
    alpha[:, -6:] = 0
    alpha[alpha < 110] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 50).astype(np.uint8))
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha[labels != largest_label] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.erode(alpha, kernel, iterations=1)

    blurred_alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.minimum(alpha, blurred_alpha)

    ys, xs = np.where(alpha > 120)
    tip_x, tip_y = (int(xs[np.argmin(xs + ys * 1.15)]), int(ys[np.argmin(xs + ys * 1.15)])) if len(xs) > 0 else (0, 0)
    return bgr, alpha, tip_x, tip_y

def sort_contours_nn(contours, start_pt=(100, 150)):
    def c_center(c):
        M = cv2.moments(c)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)

    valid = [c for c in contours if cv2.arcLength(c, False) > 10]
    sorted_res = []
    if valid:
        curr = start_pt
        rem = valid[:]
        while rem:
            best_idx = 0
            best_dist = float("inf")
            for idx, c in enumerate(rem):
                pt = c_center(c)
                d = (pt[0] - curr[0]) ** 2 + (pt[1] - curr[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            chosen = rem.pop(best_idx)
            sorted_res.append(chosen)
            curr = c_center(chosen)
    return sorted_res

def extract_continuous_trajectory(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return [(WIDTH // 2, HEIGHT // 2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    title_mask = np.zeros_like(binary)
    title_mask[:95, :] = binary[:95, :]

    body_mask = np.zeros_like(binary)
    body_mask[95:, :] = binary[95:, :]

    t_cnts, _ = cv2.findContours(title_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    b_cnts, _ = cv2.findContours(body_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    t_sorted = sorted(t_cnts, key=lambda c: cv2.boundingRect(c)[0])
    b_sorted = sort_contours_nn(b_cnts, start_pt=(150, 200))

    all_contours = t_sorted + b_sorted
    trajectory = []
    for c in all_contours:
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

    m_title = np.zeros_like(binary)
    m_title[:95, :] = binary[:95, :]

    m_left = np.zeros_like(binary)
    m_left[95:HEIGHT, :int(WIDTH * 0.48)] = binary[95:HEIGHT, :int(WIDTH * 0.48)]

    m_right_char = np.zeros_like(binary)
    m_right_char[95:int(HEIGHT * 0.72), int(WIDTH * 0.48):] = binary[95:int(HEIGHT * 0.72), int(WIDTH * 0.48):]

    m_bottom_badge = np.zeros_like(binary)
    m_bottom_badge[int(HEIGHT * 0.72):, int(WIDTH * 0.48):] = binary[int(HEIGHT * 0.72):, int(WIDTH * 0.48):]

    t_cnts, _ = cv2.findContours(m_title, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    l_cnts, _ = cv2.findContours(m_left, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    r_cnts, _ = cv2.findContours(m_right_char, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    b_cnts, _ = cv2.findContours(m_bottom_badge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    t_sorted = sorted(t_cnts, key=lambda c: cv2.boundingRect(c)[0])
    l_sorted = sort_contours_nn(l_cnts, start_pt=(150, 250))
    r_sorted = sort_contours_nn(r_cnts, start_pt=(int(WIDTH * 0.7), 250))
    b_sorted = sort_contours_nn(b_cnts, start_pt=(int(WIDTH * 0.75), int(HEIGHT * 0.85)))

    def to_points(contours):
        pts = []
        for c in contours:
            for p in c.reshape(-1, 2)[::3]:
                pts.append((int(p[0]), int(p[1])))
        return pts

    zones = [
        to_points(t_sorted),
        to_points(l_sorted),
        to_points(r_sorted),
        to_points(b_sorted),
    ]
    return [z for z in zones if len(z) > 10]

def paste_hand(frame_bgr, hand_bgr, hand_alpha, x, y):
    fh, fw = frame_bgr.shape[:2]
    hh, hw = hand_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + hw), min(fh, y + hh)
    if x1 >= x2 or y1 >= y2:
        return
    hx1, hy1 = x1 - x, y1 - y
    hx2, hy2 = hx1 + (x2 - x1), hy1 + (y2 - y1)

    sub_hand = hand_bgr[hy1:hy2, hx1:hx2]
    sub_alpha = (hand_alpha[hy1:hy2, hx1:hx2].astype(np.float32) / 255.0)[:, :, None]
    roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
    blended = sub_hand.astype(np.float32) * sub_alpha + roi * (1.0 - sub_alpha)
    frame_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

def ease_in_out(t):
    return 0.5 * (1.0 - math.cos(math.pi * t))

# ============================================================
# CAMERA MOTION PRESETS
# ============================================================
def get_motion_keyframes(motion):
    """Trả về list keyframes: (t, scale, cx, cy). t từ 0.0 → 1.0."""
    presets = {
        "zoom_in_center": [
            (0.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
            (1.0, 1.28, WIDTH * 0.50, HEIGHT * 0.50),
        ],
        "zoom_out_center": [
            (0.0, 1.28, WIDTH * 0.50, HEIGHT * 0.50),
            (1.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
        ],
        "pan_left_to_right": [
            (0.0, 1.18, WIDTH * 0.30, HEIGHT * 0.50),
            (1.0, 1.18, WIDTH * 0.70, HEIGHT * 0.50),
        ],
        "pan_right_to_left": [
            (0.0, 1.18, WIDTH * 0.70, HEIGHT * 0.50),
            (1.0, 1.18, WIDTH * 0.30, HEIGHT * 0.50),
        ],
        "zoom_in_top_left": [
            (0.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
            (1.0, 1.30, WIDTH * 0.28, HEIGHT * 0.28),
        ],
        "zoom_in_bottom_right": [
            (0.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
            (1.0, 1.30, WIDTH * 0.72, HEIGHT * 0.72),
        ],
        "ken_burns_slow": [
            (0.0, 1.08, WIDTH * 0.42, HEIGHT * 0.44),
            (1.0, 1.24, WIDTH * 0.58, HEIGHT * 0.56),
        ],
        "static": [
            (0.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
            (1.0, 1.00, WIDTH * 0.50, HEIGHT * 0.50),
        ],
    }
    return presets.get(motion, presets["zoom_in_center"])

def interpolate_motion(keyframes, p):
    """Nội suy scale, cx, cy tại tiến độ p (0..1)."""
    if p <= keyframes[0][0]:
        _, s, cx, cy = keyframes[0]
        return s, cx, cy
    if p >= keyframes[-1][0]:
        _, s, cx, cy = keyframes[-1]
        return s, cx, cy
    for i in range(len(keyframes) - 1):
        k0, k1 = keyframes[i], keyframes[i + 1]
        if k0[0] <= p <= k1[0]:
            span = k1[0] - k0[0]
            local = (p - k0[0]) / span if span > 0 else 1.0
            eased = ease_in_out(local)
            s = k0[1] + (k1[1] - k0[1]) * eased
            cx = k0[2] + (k1[2] - k0[2]) * eased
            cy = k0[3] + (k1[3] - k0[3]) * eased
            return s, cx, cy
    _, s, cx, cy = keyframes[-1]
    return s, cx, cy

def crop_with_motion(frame_bgr, scale, cx, cy):
    """Crop và resize frame theo scale + tâm (cx, cy), có clamp."""
    crop_w = int(WIDTH / scale)
    crop_h = int(HEIGHT / scale)
    x1 = max(0, min(WIDTH - crop_w, int(cx - crop_w / 2)))
    y1 = max(0, min(HEIGHT - crop_h, int(cy - crop_h / 2)))
    crop = frame_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

# ============================================================
# 4 HÀM RENDER SCENE (NÂNG CẤP CAMERA MOTION)
# ============================================================
def render_scene_kttv_v2(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_duration = max(2.0, min(duration - 1.5, duration * 0.75))
    draw_frames = int(draw_duration * FPS)
    retract_frames = int(0.5 * FPS)

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise RuntimeError(f"Không đọc được file ảnh: {image_path}")
    original_bgr = cv2.resize(original_bgr, (WIDTH, HEIGHT))
    white_canvas = np.full_like(original_bgr, 255)
    reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    zone_trajectories = extract_staggered_trajectories(image_path)
    all_points = [p for z in zone_trajectories for p in z]
    if not all_points:
        all_points = [(WIDTH // 2, HEIGHT // 2)]

    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path, target_width=320)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    last_tip = all_points[0]
    smooth_cx = float(last_tip[0])
    motion_kfs = get_motion_keyframes(motion)

    for f_idx in range(total_frames):
        hand_visible = False
        hand_pos_x, hand_pos_y = 0, 0

        if f_idx < draw_frames:
            curr_idx = int((f_idx + 1) / draw_frames * len(all_points))
            prev_idx = int(f_idx / draw_frames * len(all_points))
            step_pts = all_points[prev_idx:curr_idx]

            for pt in step_pts:
                cv2.circle(reveal_mask, pt, 24, 255, -1)

            target_pt = step_pts[-1] if step_pts else all_points[min(curr_idx, len(all_points) - 1)]
            jitter_x = int(1.2 * math.sin(f_idx * 1.8))
            jitter_y = int(1.2 * math.cos(f_idx * 1.8))
            hand_pos_x = target_pt[0] + jitter_x
            hand_pos_y = target_pt[1] + jitter_y
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (HEIGHT + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame_world = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame_world, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        # Camera motion: áp dụng sau khi vẽ xong
        if f_idx < draw_frames:
            # Trong lúc vẽ, giữ camera zoom nhẹ để xem nét vẽ
            scale = 1.20
            smooth_cx = smooth_cx * 0.94 + hand_pos_x * 0.06
            cy_use = HEIGHT * 0.5
            cx_use = smooth_cx
        else:
            # Sau khi vẽ xong, áp dụng camera motion preset
            out_prog = (f_idx - draw_frames) / max(1, total_frames - draw_frames)
            s_target, cx_target, cy_target = interpolate_motion(motion_kfs, out_prog)
            # Nội suy từ trạng thái vẽ (scale 1.20, cx=smooth_cx) về trạng thái motion
            blend = ease_in_out(min(1.0, out_prog * 1.5))
            scale = 1.20 + (s_target - 1.20) * blend
            cx_use = smooth_cx + (cx_target - smooth_cx) * blend
            cy_use = HEIGHT * 0.5 + (cy_target - HEIGHT * 0.5) * blend

        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        x1 = max(0, min(WIDTH - crop_w, int(cx_use - crop_w / 2)))
        y1 = max(0, min(HEIGHT - crop_h, int(cy_use - crop_h / 2)))

        crop = frame_world[y1:y1 + crop_h, x1:x1 + crop_w]
        frame_out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame_out.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại (Chế độ 1)")

def render_scene_hybrid(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_duration = max(2.0, min(duration - 1.5, duration * 0.72))
    draw_frames = int(draw_duration * FPS)
    retract_frames = int(0.5 * FPS)

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise RuntimeError(f"Không đọc được file ảnh: {image_path}")
    original_bgr = cv2.resize(original_bgr, (WIDTH, HEIGHT))
    white_canvas = np.full_like(original_bgr, 255)
    reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    trajectory = extract_continuous_trajectory(image_path)
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path, target_width=320)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    last_tip = trajectory[0] if trajectory else (WIDTH // 2, HEIGHT // 2)
    smooth_cx, smooth_cy = float(last_tip[0]), float(last_tip[1])
    motion_kfs = get_motion_keyframes(motion)

    for f_idx in range(total_frames):
        hand_visible = False
        hand_pos_x, hand_pos_y = 0, 0

        if f_idx < draw_frames:
            curr_idx = int((f_idx + 1) / draw_frames * len(trajectory))
            prev_idx = int(f_idx / draw_frames * len(trajectory))
            step_pts = trajectory[prev_idx:curr_idx]

            for pt in step_pts:
                cv2.circle(reveal_mask, pt, 24, 255, -1)

            target_pt = step_pts[-1] if step_pts else trajectory[min(curr_idx, len(trajectory) - 1)]
            jitter_x = int(1.2 * math.sin(f_idx * 1.8))
            jitter_y = int(1.2 * math.cos(f_idx * 1.8))
            hand_pos_x = target_pt[0] + jitter_x
            hand_pos_y = target_pt[1] + jitter_y
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (HEIGHT + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame_world = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame_world, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        if f_idx < draw_frames:
            scale = 1.20
            smooth_cx = smooth_cx * 0.95 + hand_pos_x * 0.05
            smooth_cy = smooth_cy * 0.95 + hand_pos_y * 0.05
            cx_use, cy_use = smooth_cx, smooth_cy
        else:
            out_prog = (f_idx - draw_frames) / max(1, total_frames - draw_frames)
            s_target, cx_target, cy_target = interpolate_motion(motion_kfs, out_prog)
            blend = ease_in_out(min(1.0, out_prog * 1.5))
            scale = 1.20 + (s_target - 1.20) * blend
            cx_use = smooth_cx + (cx_target - smooth_cx) * blend
            cy_use = smooth_cy + (cy_target - smooth_cy) * blend

        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        x1 = max(0, min(WIDTH - crop_w, int(cx_use - crop_w / 2)))
        y1 = max(0, min(HEIGHT - crop_h, int(cy_use - crop_h / 2)))

        crop = frame_world[y1:y1 + crop_h, x1:x1 + crop_w]
        frame_out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame_out.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại (Chế độ 2)")

def render_scene_kttv_pure(image_path, duration, output_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    original_bgr = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    motion_kfs = get_motion_keyframes(motion)

    for f_idx in range(total_frames):
        p = f_idx / max(1, total_frames - 1)
        scale, cx, cy = interpolate_motion(motion_kfs, p)
        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        x1 = max(0, min(WIDTH - crop_w, int(cx - crop_w / 2)))
        y1 = max(0, min(HEIGHT - crop_h, int(cy - crop_h / 2)))
        crop = original_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
        frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

def render_scene_classic_hand(image_path, duration, output_path, hand_path, motion="zoom_in_center"):
    total_frames = max(1, round(duration * FPS))
    draw_frames = int(max(2.0, min(duration - 1.0, duration * 0.75)) * FPS)
    retract_frames = int(0.5 * FPS)
    original_bgr = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    white_canvas = np.full_like(original_bgr, 255)
    reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    trajectory = extract_continuous_trajectory(image_path)
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    last_tip = trajectory[0] if trajectory else (WIDTH // 2, HEIGHT // 2)

    for f_idx in range(total_frames):
        hand_visible = False
        if f_idx < draw_frames:
            curr_idx = int((f_idx + 1) / draw_frames * len(trajectory))
            prev_idx = int(f_idx / draw_frames * len(trajectory))
            step_pts = trajectory[prev_idx:curr_idx]
            for pt in step_pts:
                cv2.circle(reveal_mask, pt, 24, 255, -1)
            target_pt = step_pts[-1] if step_pts else trajectory[min(curr_idx, len(trajectory) - 1)]
            jitter_x = int(1.2 * math.sin(f_idx * 1.8))
            jitter_y = int(1.2 * math.cos(f_idx * 1.8))
            hand_pos_x = target_pt[0] + jitter_x
            hand_pos_y = target_pt[1] + jitter_y
            last_tip = (hand_pos_x, hand_pos_y)
            hand_visible = True
        elif f_idx < draw_frames + retract_frames:
            reveal_mask[:, :] = 255
            prog = (f_idx - draw_frames) / max(1, retract_frames)
            hand_pos_x = int(last_tip[0] + (WIDTH + 180 - last_tip[0]) * prog)
            hand_pos_y = int(last_tip[1] + (HEIGHT + 180 - last_tip[1]) * prog)
            hand_visible = True
        else:
            reveal_mask[:, :] = 255

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

# ============================================================
# RENDER BATCH
# ============================================================
def render_batch(batch_audio, scenes, batch_dir, hand_path, style,
                 cf_account, cf_token, hf_token, freetheai_key, together_key,
                 nexa_key, agnes_key, image_timeout, flux_steps=4, progress_callback=None):
    scene_videos = []
    total = len(scenes)
    if total == 0:
        raise RuntimeError("Không có cảnh nào để render trong batch này.")

    providers = build_provider_list(cf_account, cf_token, hf_token, freetheai_key,
                                     together_key, nexa_key, agnes_key, flux_steps)
    provider_names = [p["name"] for p in providers]
    st.caption(f"🔗 Thứ tự fallback: {' → '.join(provider_names)}")

    for i, s in enumerate(scenes, 1):
        img_raw = batch_dir / f"scene_{i:03d}_raw.png"
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"

        if not img.exists():
            seed = hash(f"{s['visual_prompt']}_{i}") % (2**31)
            st.write(f"🖼️ Cảnh {i}/{total}: đang tạo ảnh...")
            data, used_provider = generate_image_with_fallback(
                s["visual_prompt"], providers, timeout=image_timeout, seed=seed)
            st.caption(f"   → Tạo bởi: **{used_provider}**")
            save_image_from_bytes(data, img_raw)
            add_comic_overlays(img_raw, s["title"], s.get("callout_type", "speech"),
                               s.get("callout_text", ""), s.get("callout_side", "right"), img)

        duration = max(1.0, float(s["end"]) - float(s["start"]))
        motion = s.get("camera_motion", "zoom_in_center")

        if "Kiến Thức Thú Vị V2" in style or "1." in style:
            render_scene_kttv_v2(img, duration, vid, hand_path, motion)
        elif "Độc bản" in style or "2." in style:
            render_scene_hybrid(img, duration, vid, hand_path, motion)
        elif "Chỉ Camera" in style or "3." in style:
            render_scene_kttv_pure(img, duration, vid, motion)
        else:
            render_scene_classic_hand(img, duration, vid, hand_path, motion)

        scene_videos.append(vid)
        if progress_callback:
            progress_callback(i / total)

    concat_file = batch_dir / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_videos), encoding="utf-8")
    batch_video = batch_dir / "batch_video.mp4"
    run_cmd([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy", "-movflags", "+faststart",
        str(batch_video),
    ], timeout=900)

    final_batch = batch_dir / "batch_final.mp4"
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(batch_video),
        "-i", str(batch_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(final_batch),
    ], timeout=900)
    return final_batch

def concat_batches(batch_videos, output_path):
    concat = output_path.parent / "batches.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in batch_videos), encoding="utf-8")
    run_cmd([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart",
        str(output_path),
    ], timeout=1800)

# ============================================================
# LUỒNG ỨNG DỤNG CHÍNH
# ============================================================
st.sidebar.divider()

if st.sidebar.button("🔎 KIỂM TRA NHÀ CUNG CẤP ẢNH", use_container_width=True):
    providers = build_provider_list(cf_account, cf_token, hf_token, freetheai_key,
                                     together_key, nexa_key, agnes_key, flux_steps)
    st.write(f"**Phát hiện {len(providers)} nhà cung cấp:**")
    for p in providers:
        st.write(f"• {p['name']}")
    try:
        with st.spinner("Đang test tạo ảnh qua waterfall..."):
            data, used = generate_image_with_fallback(
                "2D comic doodle: a man standing at the edge of a cliff at sunset, "
                "looking down at a vast ocean of papers below, red sunset, blue waves, "
                "white background, bold black outlines, no text",
                providers, timeout=120)
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((WIDTH, HEIGHT))
        st.success(f"✅ Ảnh tạo thành công qua **{used}**!")
        st.image(img, caption=f"Nhà cung cấp: {used}", use_container_width=True)
    except Exception as e:
        st.error(f"❌ {e}")

audio = st.file_uploader(
    "🎤 Tải lên tệp ghi âm giọng nói",
    type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"],
)

if audio:
    st.audio(audio)

    if st.button("🚀 BẮT ĐẦU TẠO VIDEO", type="primary", use_container_width=True):
        if not groq_key:
            st.error("Vui lòng nhập Groq API Key.")
            st.stop()

        root = Path(tempfile.mkdtemp(prefix="wb_super_"))
        try:
            source = root / audio.name
            source.write_bytes(audio.getbuffer())
            duration = ffprobe_duration(source)
            st.info(f"Thời lượng: {duration/60:.2f} phút. Chia batch 5 phút.")

            client = groq_client(groq_key)
            batch_dir = root / "batches"
            batch_dir.mkdir()
            chunks = chunk_audio(source, batch_dir)
            hand_path = Path("hand.png")

            batch_videos = []
            all_scene_count = 0
            progress = st.progress(0)
            status = st.empty()

            valid_chunks = [(bi, ch, ffprobe_duration(ch)) for bi, ch in enumerate(chunks)
                            if ffprobe_duration(ch) >= 5.0 or bi == 0]
            if not valid_chunks:
                st.error("Không có đoạn âm thanh hợp lệ.")
                st.stop()

            # Xác định camera mode
            if camera_motion_mode == "Auto (AI chọn cho từng cảnh)":
                camera_mode = "auto"
            elif camera_motion_mode == "Random (Code chọn ngẫu nhiên)":
                camera_mode = "random"
            else:
                fixed = camera_motion_mode.replace("Cố định: ", "").strip()
                camera_mode = f"fixed:{fixed}"

            for idx, (bi, chunk, bdur) in enumerate(valid_chunks):
                bstart = bi * BATCH_SECONDS
                status.write(f"🧠 Đợt {idx+1}/{len(valid_chunks)} — Nhận diện giọng nói...")
                tr = transcribe_file(client, chunk, stt_model)
                segs = normalize_segments(tr, bstart)
                batch_text = "\n".join(f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}" for x in segs)

                status.write(f"✂️ Đợt {idx+1}/{len(valid_chunks)} — Qwen lên kịch bản...")
                scenes = make_scene_plan(
                    client, batch_text, bstart, bdur, planner_model,
                    scene_min, scene_max, max_scenes, camera_mode,
                )
                st.write(f"**Đợt {idx+1}: {bdur:.1f}s → {len(scenes)} cảnh**")
                for si, s in enumerate(scenes, 1):
                    ci = f" | [{s.get('callout_type', '').upper()}]: \"{s.get('callout_text', '')}\"" if s.get('callout_text') else " | [Chỉ hình ảnh]"
                    cm = s.get("camera_motion", "zoom_in_center")
                    st.caption(f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}{ci} | 🎥 {cm}")

                batch_work = root / f"work_{idx+1:03d}"
                batch_work.mkdir()

                status.write(f"🎨 Đợt {idx+1}/{len(valid_chunks)} — Vẽ tranh + render...")
                def cb(frac, idx=idx):
                    progress.progress(min(1.0, (idx + frac) / len(valid_chunks)))

                bv = render_batch(
                    chunk, scenes, batch_work, hand_path, draw_style,
                    cf_account, cf_token, hf_token, freetheai_key,
                    together_key, nexa_key, agnes_key, image_timeout,
                    flux_steps, cb,
                )
                saved = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bv, saved)
                batch_videos.append(saved)
                all_scene_count += len(scenes)
                shutil.rmtree(batch_work, ignore_errors=True)

            progress.progress(1.0)
            status.write("🎬 Ghép video hoàn chỉnh...")
            final = root / "video_hoan_thien_final.mp4"
            concat_batches(batch_videos, final)

            st.success(f"Hoàn thành! {all_scene_count} cảnh đã được tạo.")
            st.video(str(final))
            st.download_button(
                "⬇️ TẢI VIDEO MP4",
                data=final.read_bytes(),
                file_name="video_hoan_thien_final.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        except Exception as e:
            st.exception(e)
