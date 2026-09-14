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
import urllib.parse
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
import cv2
import numpy as np

APP_TITLE = "Xưởng Video Diễn Hoạt Kiến Thức AI (Bản Hoàn Thiện)"
BATCH_SECONDS = 5 * 60  # 5 phút chuẩn mỗi đợt
FPS = 30
WIDTH = 1280
HEIGHT = 720
CLOUDFLARE_AI_URL = "https://api.cloudflare.com/client/v4/accounts/"

# -----------------------------
# Giao diện / Cấu hình
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

st.title("🎬 Xưởng Video Diễn Hoạt Kiến Thức AI")
st.caption("Qwen biên kịch Tiếng Việt tự nhiên + Đa dạng kiểu chữ Comic + Khử sạch 100% bóng mờ bàn tay + Khóa nhịp 20s/cảnh")

with st.sidebar:
    st.header("🔑 Cấu hình API")
    groq_key = st.text_input(
        "Khóa Groq API",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password",
        help="Dùng để nhận diện giọng nói và lên kịch bản phân cảnh.",
    )
    cloudflare_account_id = st.text_input(
        "Cloudflare Account ID",
        value=st.secrets.get("CLOUDFLARE_ACCOUNT_ID", os.getenv("CLOUDFLARE_ACCOUNT_ID", "")),
        type="password",
    )
    cloudflare_token = st.text_input(
        "Cloudflare Workers AI API Token",
        value=st.secrets.get("CLOUDFLARE_API_TOKEN", os.getenv("CLOUDFLARE_API_TOKEN", "")),
        type="password",
    )
    pollinations_key = st.text_input(
        "Khóa Pollinations API (Dự phòng)",
        value=st.secrets.get("POLLINATIONS_API_KEY", os.getenv("POLLINATIONS_API_KEY", "")),
        type="password",
        help="Sử dụng khi Cloudflare API gặp lỗi hoặc quá tải.",
    )

    st.header("🧠 Mô hình Groq")
    stt_model = st.selectbox(
        "Mô hình nghe giọng nói",
        ["whisper-large-v3", "whisper-large-v3-turbo"],
        index=0,
    )
    planner_model = st.selectbox(
        "Mô hình biên kịch (Khuyên dùng Qwen)",
        ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0,
    )

    st.header("🎨 Cloudflare AI")
    image_model = st.selectbox(
        "Mô hình tạo ảnh",
        ["@cf/black-forest-labs/flux-1-schnell"],
        index=0,
    )

    st.header("🎬 Phong cách diễn hoạt")
    draw_style = st.selectbox(
        "Lựa chọn phong cách dựng phim",
        [
            "1. Kiến Thức Thú Vị V2 (Vẽ tuần tự + Bong bóng thoại biến hóa + Pan/Zoom)",
            "2. Độc bản Hybrid (Tay vẽ bám nét + Camera Steadicam lướt theo bút)",
            "3. Kiến Thức Thú Vị (Chỉ Camera Pan & Zoom, ẩn bàn tay)",
            "4. Bảng trắng cổ điển (Tay vẽ góc máy tĩnh không lia)",
        ],
        index=0,
    )

    st.header("⏱️ Khóa nhịp cảnh (Chống chia vụn)")
    scene_min = st.slider("Thời lượng cảnh tối thiểu (giây)", 18, 25, 19)
    scene_max = st.slider("Thời lượng cảnh tối đa (giây)", 22, 35, 27)
    if scene_max < scene_min:
        scene_max = scene_min

    st.header("⚙️ Cài đặt khác")
    max_scenes_per_batch = st.slider("Số cảnh tối đa mỗi đợt 5 phút", 5, 20, 14)
    image_timeout = st.slider("Thời gian chờ tạo ảnh (giây)", 30, 180, 120)

# -----------------------------
# Tiện ích hệ thống
# -----------------------------
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
        str(path)
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
        pattern
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

# -----------------------------
# Bộ điều phối kịch bản
# -----------------------------
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model, min_s, max_s, max_scenes):
    expected_scenes = max(1, round(batch_duration / 23.0))

    system = f"""
Bạn là giám đốc sáng tạo kịch bản cho kênh hoạt họa kiến thức phong cách "Kiến Thức Thú Vị".
Nhiệm vụ: Chia đoạn âm thanh {batch_duration:.0f}s thành khoảng {expected_scenes} cảnh lớn, mạch lạc, không chia vụn ({min_s}-{max_s}s/cảnh).

QUY TẮC NỘI DUNG VÀ ĐA DẠNG HÓA CHỮ (RẤT QUAN TRỌNG):
1. "title": Tiêu đề tiếng Việt ngắn gọn (3-6 từ, viết hoa) tóm tắt luận điểm chính của cảnh.
2. ĐA DẠNG HÓA KIỂU CHỮ TRÊN TRANH ("callout_type" và "callout_text"):
     * "speech": Bong bóng thoại tròn ("LÃI SUẤT QUÁ CAO!", "LẠI PHẢI NHẬN À?", "KHÔNG DÁM TỪ CHỐI...")
     * "thought": Đám mây suy nghĩ ("HỌ CÓ GHÉT MÌNH KHÔNG?", "BIẾT TÍNH SAO ĐÂY?...")
     * "sticker": Nhãn dán Comic ("BẪY TÂM LÝ!", "MẤT HẾT TỰ DO!", "KIỆT SỨC!")
     * "none": Để trống (callout_text = ""), không chèn chữ gì.
3. VỊ TRÍ CHỮ ("callout_side"): Chọn "left" hoặc "right".
4. MÔ TẢ TRANH ("visual_prompt"):
   - Tiếng Anh cho FLUX, phong cách 2D comic doodle bảng trắng, nét mực đen dày rõ nét, có điểm nhấn màu đỏ/xanh.
   - Nhân vật vẽ bán thân (waist-up) biểu cảm lo âu.
   - CẤM TIỆT VẼ CHỮ, CẤM VẼ BẢNG TRẮNG CÓ KHAY BÚT, CẤM VẼ BONG BÓNG THOẠI RỖNG.
5. Trả về đúng JSON.

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0,
      "end": 22.0,
      "title": "CỐ GẮNG HÀI LÒNG MỌI NGƯỜI",
      "callout_type": "speech",
      "callout_text": "KHỔ QUÁ RỒI!",
      "callout_side": "right",
      "visual_prompt": "16:9 whiteboard comic doodle: left side has many demanding hands reaching in with papers, right side has a waist-up cartoon man sweating and overwhelmed, clean white background, bold black lines, selective red color accents, strictly zero text, no physical board frame"
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

    clean = []
    for s in raw_scenes[:max_scenes]:
        try:
            a = max(0.0, float(s["start"]))
            b = min(batch_duration, float(s["end"]))
            if b <= a + 1.0:
                continue
            clean.append({
                "start": a,
                "end": b,
                "title": str(s.get("title", "BÀI HỌC KIẾN THỨC")).strip().upper(),
                "callout_type": str(s.get("callout_type", "speech")).strip().lower(),
                "callout_text": str(s.get("callout_text", "")).strip(),
                "callout_side": str(s.get("callout_side", "right")).strip().lower(),
                "visual_prompt": sanitize_prompt_text(str(s.get("visual_prompt", ""))),
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
            "visual_prompt": "A 16:9 educational whiteboard comic: left side has contract papers with downward trend arrow, right side has a waist-up comic man sweating and stressed, bold black lines, selective red color, pure white background, no text",
        }]

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
                "visual_prompt": s["visual_prompt"],
            })
            final_scenes.append({
                "start": mid,
                "end": s["end"],
                "title": f"{s['title']} (TIẾP)",
                "callout_type": "sticker",
                "callout_text": "CẦN CẨN TRỌNG!",
                "callout_side": "right",
                "visual_prompt": s["visual_prompt"] + ", continuation scene, waist-up character, clean white background",
            })
        else:
            final_scenes.append(s)

    return final_scenes

# -----------------------------
# Engine Tạo Ảnh: Cloudflare AI & Pollinations Fallback
# -----------------------------
def build_full_prompt(prompt):
    safe_prompt = sanitize_prompt_text(prompt)
    return f"""
Comprehensive 16:9 widescreen educational whiteboard comic illustration of {safe_prompt}.
CRITICAL STYLE REQUIREMENTS:
- Authentic 2D comic doodle art style, thick black ink contour outlines, expressive cartoon characters with vivid facial expressions, waist-up shot or sitting at desk. NO awkward stick-legs.
- Balanced layout: subject matter on left, character on right, leaving clean open white space for overlays.
- PURE SOLID FLAT WHITE PAPER BACKGROUND.
- ABSOLUTELY NO physical whiteboard frame, NO aluminum borders, NO pen tray, NO markers on table, NO eraser.
- Selective vibrant red and green/blue spot colors on key elements.
- STRICTLY WORDLESS: Absolutely NO English words, NO Vietnamese words, NO letters, NO numbers, NO captions, NO empty speech balloons, NO calendars.
"""

def cloudflare_image_request(prompt, account_id, api_token, model, timeout=120):
    account_id = (account_id or "").strip()
    api_token = (api_token or "").strip()
    if not account_id or not api_token:
        raise RuntimeError("Chưa cấu hình Cloudflare Account ID hoặc API Token.")

    url = f"{CLOUDFLARE_AI_URL}{account_id}/ai/run/{model}"
    payload = {"prompt": build_full_prompt(prompt), "steps": 4}
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 401:
                raise RuntimeError("Cloudflare 401: API Token không hợp lệ.")
            if response.status_code == 403:
                raise RuntimeError("Cloudflare 403: Không có quyền gọi mô hình này.")
            if response.status_code == 429:
                time.sleep(3 * attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Cloudflare HTTP {response.status_code}: {response.text[:500]}")

            data = response.json()
            if not data.get("success", True):
                raise RuntimeError(f"Cloudflare AI lỗi: {data}")

            result = data.get("result", {})
            image_b64 = result.get("image")
            if not image_b64:
                raise RuntimeError("Cloudflare không trả về dữ liệu ảnh Base64.")

            return base64.b64decode(image_b64)
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise last_error

def pollinations_image_request(prompt, api_key="", timeout=120):
    full_prompt = build_full_prompt(prompt)
    # Mã hóa triệt để ký tự đặc biệt (kể cả dấu / với safe="") để tránh lỗi 404 URL
    encoded_prompt = urllib.parse.quote(full_prompt, safe="")
    
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={WIDTH}&height={HEIGHT}&nologo=true&seed={int(time.time())}"
    if api_key:
        url += f"&key={api_key.strip()}"
        
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    
    if response.status_code != 200:
        raise RuntimeError(f"Pollinations HTTP {response.status_code}: {response.text[:300]}")
    if not response.content:
        raise RuntimeError("Pollinations không trả về dữ liệu ảnh.")
        
    return response.content

def cloudflare_image(prompt, account_id, api_token, model, output_path, timeout=120):
    data = cloudflare_image_request(prompt, account_id, api_token, model, timeout)
    save_and_process_image(data, output_path)

def generate_image_with_fallback(prompt, account_id, api_token, model, pol_key, output_path, timeout=120):
    try:
        # 1. Thử tạo ảnh bằng Cloudflare API
        cloudflare_image(prompt, account_id, api_token, model, output_path, timeout)
    except Exception as cf_err:
        # 2. Nếu Cloudflare lỗi, tự động Fallback sang Pollinations API
        try:
            st.warning(f"⚠️ Cloudflare AI lỗi ({cf_err}). Đang chuyển sang Pollinations API...")
            data = pollinations_image_request(prompt, api_key=pol_key, timeout=timeout)
            save_and_process_image(data, output_path)
        except Exception as pol_err:
            raise RuntimeError(f"Cả Cloudflare API ({cf_err}) và Pollinations API ({pol_err}) đều thất bại!")

def save_and_process_image(image_bytes, output_path):
    Path(output_path).write_bytes(image_bytes)
    try:
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            im = im.convert("RGB")
            im = im.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            im.save(output_path, "JPEG", quality=95)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Dữ liệu ảnh không hợp lệ: {e}")

def test_cloudflare_api(account_id, api_token, model, timeout=120):
    data = cloudflare_image_request(
        prompt="A 16:9 whiteboard comic: left side has contract papers with falling red arrow, right side has a waist-up comic man sweating and stressed at desk, bold black lines, selective red color, pure white background, zero text, no pen tray",
        account_id=account_id,
        api_token=api_token,
        model=model,
        timeout=timeout,
    )
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

def font_for(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

# -----------------------------
# Bộ máy Đa dạng hóa Chữ Comic Tiếng Việt
# -----------------------------
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

# -----------------------------
# Bộ máy Bàn tay & Khử bóng mờ
# -----------------------------
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

# -----------------------------
# PHONG CÁCH 1: Kiến Thức Thú Vị V2
# -----------------------------
def render_scene_kttv_v2(image_path, duration, output_path, hand_path):
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
            hand_visible = False

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame_world = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame_world, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        if f_idx < draw_frames:
            scale = 1.20
            smooth_cx = smooth_cx * 0.94 + hand_pos_x * 0.06
        else:
            out_prog = ease_in_out((f_idx - draw_frames) / max(1, total_frames - draw_frames))
            scale = 1.20 - 0.20 * out_prog
            smooth_cx = smooth_cx * (1.0 - out_prog) + (WIDTH * 0.5) * out_prog

        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        half_w = crop_w // 2

        x1 = max(0, min(WIDTH - crop_w, int(smooth_cx - half_w)))
        y1 = 0

        crop = frame_world[y1:y1 + crop_h, x1:x1 + crop_w]
        frame_out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame_out.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại trong Chế độ 1.")

# -----------------------------
# PHONG CÁCH 2: Độc Bản Hybrid
# -----------------------------
def render_scene_hybrid(image_path, duration, output_path, hand_path):
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
            hand_visible = False

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame_world = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame_world, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        if f_idx < draw_frames:
            scale = 1.20
            smooth_cx = smooth_cx * 0.95 + hand_pos_x * 0.05
            smooth_cy = smooth_cy * 0.95 + hand_pos_y * 0.05
        else:
            out_prog = ease_in_out((f_idx - draw_frames) / max(1, total_frames - draw_frames))
            scale = 1.20 - 0.20 * out_prog
            smooth_cx = smooth_cx * (1.0 - out_prog) + (WIDTH * 0.5) * out_prog
            smooth_cy = smooth_cy * (1.0 - out_prog) + (HEIGHT * 0.5) * out_prog

        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        half_w, half_h = crop_w // 2, crop_h // 2
        clamped_cx = max(half_w, min(WIDTH - half_w, int(smooth_cx)))
        clamped_cy = max(half_h, min(HEIGHT - half_h, int(smooth_cy)))

        crop = frame_world[clamped_cy - half_h:clamped_cy + half_h, clamped_cx - half_w:clamped_cx + half_w]
        frame_out = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame_out.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg render thất bại trong Chế độ 2.")

# -----------------------------
# PHONG CÁCH 3: Kiến Thức Thú Vị Pure
# -----------------------------
def render_scene_kttv_pure(image_path, duration, output_path):
    total_frames = max(1, round(duration * FPS))
    original_bgr = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    pt_left = (WIDTH * 0.35, HEIGHT * 0.50)
    pt_right = (WIDTH * 0.68, HEIGHT * 0.50)
    pt_center = (WIDTH * 0.50, HEIGHT * 0.50)

    for f_idx in range(total_frames):
        p = f_idx / max(1, total_frames - 1)
        if p < 0.35:
            scale = 1.25
            cx = pt_left[0]
        elif p < 0.65:
            sub_p = ease_in_out((p - 0.35) / 0.30)
            scale = 1.25
            cx = pt_left[0] + (pt_right[0] - pt_left[0]) * sub_p
        elif p < 0.88:
            sub_p = ease_in_out((p - 0.65) / 0.23)
            scale = 1.25 - 0.25 * sub_p
            cx = pt_right[0] + (pt_center[0] - pt_right[0]) * sub_p
        else:
            scale = 1.0
            cx = pt_center[0]

        crop_w = int(WIDTH / scale)
        crop_h = int(HEIGHT / scale)
        x1 = max(0, min(WIDTH - crop_w, int(cx - crop_w / 2)))
        y1 = 0

        crop = original_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
        frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

# -----------------------------
# PHONG CÁCH 4: Bảng trắng cổ điển
# -----------------------------
def render_scene_classic_hand(image_path, duration, output_path, hand_path):
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
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)
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
            hand_visible = False

        blur = cv2.GaussianBlur(reveal_mask, (13, 13), 0)
        alpha = (blur.astype(np.float32) / 255.0)[:, :, None]
        frame = (original_bgr * alpha + white_canvas * (1.0 - alpha)).astype(np.uint8)

        if hand_visible:
            paste_hand(frame, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

# -----------------------------
# Quy trình render theo đợt
# -----------------------------
def render_batch(batch_audio, scenes, batch_dir, hand_path, style, pol_key="", progress_callback=None):
    scene_videos = []
    total = len(scenes)
    for i, s in enumerate(scenes, 1):
        img_raw = batch_dir / f"scene_{i:03d}_raw.png"
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"
        
        if not img.exists():
            # Tự động gọi Fallback giữa Cloudflare & Pollinations
            generate_image_with_fallback(
                s["visual_prompt"],
                cloudflare_account_id,
                cloudflare_token,
                image_model,
                pol_key,
                img_raw,
                image_timeout
            )
            add_comic_overlays(img_raw, s["title"], s.get("callout_type", "speech"), s.get("callout_text", ""), s.get("callout_side", "right"), img)

        duration = max(1.0, float(s["end"]) - float(s["start"]))

        if "Kiến Thức Thú Vị V2" in style or "1." in style:
            render_scene_kttv_v2(img, duration, vid, hand_path)
        elif "Độc bản" in style or "2." in style:
            render_scene_hybrid(img, duration, vid, hand_path)
        elif "Chỉ Camera" in style or "3." in style:
            render_scene_kttv_pure(img, duration, vid)
        else:
            render_scene_classic_hand(img, duration, vid, hand_path)

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
        str(batch_video)
    ], timeout=900)

    final_batch = batch_dir / "batch_final.mp4"
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(batch_video),
        "-i", str(batch_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(final_batch)
    ], timeout=900)
    return final_batch

def concat_batches(batch_videos, output_path):
    concat = output_path.parent / "batches.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in batch_videos), encoding="utf-8")
    run_cmd([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart",
        str(output_path)
    ], timeout=1800)

# -----------------------------
# Luồng ứng dụng chính
# -----------------------------
st.sidebar.divider()
if st.sidebar.button("🔎 KIỂM TRA ẢNH MẪU COMIC", use_container_width=True):
    try:
        with st.spinner("Cloudflare đang vẽ tranh minh họa..."):
            test_img = test_cloudflare_api(cloudflare_account_id, cloudflare_token, image_model, 120)
        st.success("✅ Ảnh tạo thành công — Bố cục rộng, sạch chữ rác, sẵn sàng ghép thoại!")
        st.image(test_img, caption="Ảnh mẫu Comic Doodle (Chuẩn Kiến Thức Thú Vị)", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")

audio = st.file_uploader("🎤 Tải lên tệp ghi âm giọng nói", type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"])

if audio:
    st.audio(audio)

    if st.button("🚀 BẮT ĐẦU TẠO VIDEO V2", type="primary", use_container_width=True):
        if not groq_key or not cloudflare_account_id or not cloudflare_token:
            st.error("Vui lòng nhập đầy đủ Groq API Key, Cloudflare Account ID và Token.")
            st.stop()

        root = Path(tempfile.mkdtemp(prefix="wb_final_fix_"))
        try:
            source = root / audio.name
            source.write_bytes(audio.getbuffer())
            duration = ffprobe_duration(source)
            st.info(f"Thời lượng âm thanh: {duration/60:.2f} phút. Hệ thống chia batch 5 phút chuẩn.")

            client = groq_client(groq_key)
            batch_dir = root / "batches"
            batch_dir.mkdir()
            chunks = chunk_audio(source, batch_dir)
            hand_path = Path("hand.png")

            batch_videos = []
            all_scene_count = 0
            progress = st.progress(0)
            status = st.empty()

            valid_chunks = [(bi, ch, ffprobe_duration(ch)) for bi, ch in enumerate(chunks) if ffprobe_duration(ch) >= 5.0 or bi == 0]

            for idx, (bi, chunk, bdur) in enumerate(valid_chunks):
                bstart = bi * BATCH_SECONDS
                status.write(f"🧠 Đợt {idx+1}/{len(valid_chunks)} — Đang nhận diện giọng nói...")
                tr = transcribe_file(client, chunk, stt_model)
                segs = normalize_segments(tr, bstart)
                batch_text = "\n".join(f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}" for x in segs)

                status.write(f"✂️ Đợt {idx+1}/{len(valid_chunks)} — Qwen đang lên kịch bản chuẩn nhịp 20-25s...")
                scenes = make_scene_plan(client, batch_text, bstart, bdur, planner_model, scene_min, scene_max, max_scenes_per_batch)

                st.write(f"**Đợt {idx+1}: {bdur:.1f}s → {len(scenes)} cảnh (Đã chuẩn hóa nhịp điệu)**")
                for si, s in enumerate(scenes, 1):
                    callout_info = f" | [{s.get('callout_type', '').upper()}]: \"{s.get('callout_text', '')}\"" if s.get('callout_text') else " | [Chỉ hình ảnh]"
                    st.caption(f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}{callout_info}")

                batch_work = root / f"work_{idx+1:03d}"
                batch_work.mkdir()

                status.write(f"🎨 Đợt {idx+1}/{len(valid_chunks)} — Đang vẽ tranh và render diễn hoạt...")
                def cb(frac, idx=idx):
                    progress.progress(min(1.0, (idx + frac) / len(valid_chunks)))

                bv = render_batch(chunk, scenes, batch_work, hand_path, draw_style, pollinations_key, cb)
                saved_batch = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bv, saved_batch)
                batch_videos.append(saved_batch)
                all_scene_count += len(scenes)
                shutil.rmtree(batch_work, ignore_errors=True)

            progress.progress(1.0)
            status.write("🎬 Đang kết hợp video hoàn chỉnh...")
            final = root / "video_hoan_thien_final.mp4"
            concat_batches(batch_videos, final)

            st.success(f"Hoàn thành xuất sắc! Đã tạo {all_scene_count} cảnh chuẩn nhịp Kiến Thức Thú Vị (20-25s/cảnh).")
            st.video(str(final))
            st.download_button(
                "⬇️ TẢI VIDEO MP4 VỀ MÁY",
                data=final.read_bytes(),
                file_name="video_hoan_thien_final.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)
        finally:
            pass
