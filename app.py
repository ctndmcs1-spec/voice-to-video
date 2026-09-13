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

APP_TITLE = "Xưởng Video Diễn Hoạt Độc Bản AI"
BATCH_SECONDS = 5 * 60  # 5 phút chuẩn
FPS = 30
WIDTH = 1280
HEIGHT = 720
CLOUDFLARE_AI_URL = "https://api.cloudflare.com/client/v4/accounts/"

# -----------------------------
# Giao diện / Cấu hình
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="✏️", layout="wide")

st.title("✏️ Xưởng Tạo Video Diễn Hoạt Độc Bản AI")
st.caption("Bàn tay vẽ nét thực tế + Camera Steadicam lướt bám theo ngòi bút + Zoom out toàn cảnh")

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

    st.header("🧠 Mô hình Groq")
    stt_model = st.selectbox(
        "Mô hình nghe giọng nói",
        ["whisper-large-v3", "whisper-large-v3-turbo"],
        index=0,
    )
    planner_model = st.selectbox(
        "Mô hình biên kịch",
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        index=0,
    )

    st.header("🎨 Cloudflare AI")
    image_model = st.selectbox(
        "Mô hình tạo ảnh",
        ["@cf/black-forest-labs/flux-1-schnell"],
        index=0,
    )

    st.header("🎬 Hiệu ứng diễn hoạt (Signature Style)")
    draw_style = st.selectbox(
        "Phong cách dựng phim",
        [
            "Độc bản: Bàn tay vẽ + Camera lướt theo bút + Zoom out",
            "Kiến Thức Thú Vị (Chỉ lia máy + Zoom động, ẩn tay)",
            "Bảng trắng cổ điển (Góc máy tĩnh không lia)",
        ],
        index=0,
    )

    scene_min = st.slider("Thời lượng cảnh tối thiểu (giây)", 15, 25, 15)
    scene_max = st.slider("Thời lượng cảnh tối đa (giây)", 20, 30, 30)
    if scene_max < scene_min:
        scene_max = scene_min

    st.header("⚙️ Cài đặt khác")
    max_scenes_per_batch = st.slider("Số cảnh tối đa mỗi đợt 5 phút", 5, 25, 20)
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
# Lập kịch bản phân cảnh (Khôi phục logic bù khoảng lặng + Khóa chặn >35s)
# -----------------------------
def make_scene_plan(client, transcript_text, batch_start, batch_duration, model, min_s, max_s, max_scenes):
    system = f"""
You are the visual director for a signature animated whiteboard channel.

Task:
Turn the voice transcript into a rich, 3-part storyboard scene.

HARD RULES:
1. Each scene must be {min_s}-{max_s} seconds (aim for around 18-24s).
2. COMPREHENSIVE 3-PART STORYBOARD: Divide canvas into 3 connected zones on a 16:9 layout:
   - LEFT: Trigger / context / origin of problem.
   - CENTER: Main character (STRICTLY WAIST-UP half body or sitting behind a desk with deep facial emotions. NO awkward floating legs).
   - RIGHT: Outcome / consequence / metaphor icons.
   - Curved doodle arrows connecting all zones.
3. COLOR ACCENTS: Bold black outlines on pure white background, selective red and blue spot colors.
4. STRICTLY ZERO TEXT: Absolutely NO words or letters. Use visual metaphor icons (?, !, ⚠️, ❌, ⬇️, 💔) instead of speech bubbles.
5. Safety: NEVER use words like blood, kill, suicide, weapon.
6. Return ONLY valid JSON.

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0,
      "end": 20.0,
      "title": "tiêu đề tiếng Việt ngắn gọn",
      "summary": "tóm tắt tiếng Việt",
      "visual_prompt": "detailed 3-part scene description in English, waist-up character, rich metaphors, zero text"
    }}
  ]
}}
"""
    user = f"""
Batch begins at absolute time {batch_start:.2f}s.
Batch duration: {batch_duration:.2f}s.

TRANSCRIPT:
{transcript_text}
"""
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.2,
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
                "title": str(s.get("title", "Cảnh")),
                "summary": str(s.get("summary", "")),
                "visual_prompt": sanitize_prompt_text(str(s.get("visual_prompt", ""))),
            })
        except Exception:
            continue

    if not clean:
        clean = [{
            "start": 0.0,
            "end": batch_duration,
            "title": "Tổng kết nội dung",
            "summary": (transcript_text[:120] if transcript_text else "Kết thúc nội dung"),
            "visual_prompt": "A comprehensive 3-part whiteboard infographic with a waist-up expressive character sitting at a desk in center, question marks on left, insight lightbulb on right, pure white background",
        }]

    # 1. Cảnh đầu tiên luôn bắt đầu từ 0.0s
    clean[0]["start"] = 0.0

    # 2. Lấp khoảng trống: Kéo dài ảnh cảnh trước để phủ kín khoảng lặng tới sát cảnh sau
    for i in range(len(clean) - 1):
        next_start = clean[i + 1]["start"]
        if clean[i]["end"] < next_start:
            clean[i]["end"] = next_start
        elif clean[i + 1]["start"] < clean[i]["end"]:
            clean[i + 1]["start"] = clean[i]["end"]

    # 3. Xử lý đoạn đuôi của Batch để không bị hụt dù chỉ 1 giây âm thanh
    if clean[-1]["end"] < batch_duration:
        rem = batch_duration - clean[-1]["end"]
        if rem <= 30.0:
            clean[-1]["end"] = batch_duration
        else:
            curr = clean[-1]["end"]
            step_idx = 1
            while batch_duration - curr > 0:
                r_dur = batch_duration - curr
                step = min(25.0, r_dur) if r_dur > 30.0 else r_dur
                clean.append({
                    "start": curr,
                    "end": curr + step,
                    "title": f"Cảnh báo & Lời kết {step_idx}",
                    "summary": "Tổng kết nội dung bài học",
                    "visual_prompt": "A comprehensive 3-part whiteboard infographic of a waist-up person making a wise choice, warning signpost, light ahead, clean white background",
                })
                curr += step
                step_idx += 1

    # 4. KHÓA CHẶN AN TOÀN: Tuyệt đối không để cảnh nào dài hơn 35s (tự động chẻ đôi cảnh)
    final_scenes = []
    for s in clean:
        dur = s["end"] - s["start"]
        if dur > 35.0:
            mid = s["start"] + dur / 2.0
            final_scenes.append({
                "start": s["start"],
                "end": mid,
                "title": s["title"],
                "summary": s["summary"],
                "visual_prompt": s["visual_prompt"],
            })
            final_scenes.append({
                "start": mid,
                "end": s["end"],
                "title": f"{s['title']} (tiếp)",
                "summary": s["summary"],
                "visual_prompt": s["visual_prompt"] + ", continuation part, waist-up, clean white background",
            })
        else:
            final_scenes.append(s)

    return final_scenes

# -----------------------------
# Cloudflare Workers AI Engine
# -----------------------------
def cloudflare_image_request(prompt, account_id, api_token, model, timeout=120):
    account_id = (account_id or "").strip()
    api_token = (api_token or "").strip()
    if not account_id:
        raise RuntimeError("Chưa nhập Cloudflare Account ID.")
    if not api_token:
        raise RuntimeError("Chưa nhập Cloudflare Workers AI API Token.")

    url = f"{CLOUDFLARE_AI_URL}{account_id}/ai/run/{model}"
    safe_prompt = sanitize_prompt_text(prompt)

    full_prompt = f"""
Comprehensive 16:9 widescreen educational whiteboard comic infographic of {safe_prompt}.
VISUAL RULES:
- Full landscape mindmap layout divided into 3 connected zones (Left context, Center character, Right consequence).
- Character must be WAIST-UP half-body or sitting behind a desk, highly expressive cartoon emotion, thick bold ink outlines. NO floating leg doodles.
- Pure solid white paper background.
- Selective bright red and blue spot color fills on key metaphor icons.
- STRICTLY WORDLESS: Absolutely NO words, NO letters, NO text, NO typography, NO captions. Universal icons only (?, !, ⚠️, ❌).
"""
    payload = {
        "prompt": full_prompt,
        "steps": 4,
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

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

            image_bytes = base64.b64decode(image_b64)
            return image_bytes
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise last_error

def cloudflare_image(prompt, account_id, api_token, model, output_path, timeout=120):
    data = cloudflare_image_request(
        prompt=prompt,
        account_id=account_id,
        api_token=api_token,
        model=model,
        timeout=timeout,
    )
    Path(output_path).write_bytes(data)
    try:
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            im = im.convert("RGB")
            im = im.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
            im.save(output_path, "JPEG", quality=95)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"Ảnh Cloudflare không hợp lệ: {e}")

def test_cloudflare_api(account_id, api_token, model, timeout=120):
    data = cloudflare_image_request(
        prompt="A 3-part comprehensive mindmap: on the left a boss giving tasks, in the center a waist-up stressed character holding head at desk, on the right an empty battery icon, bold ink, red spot colors, wordless",
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

def add_title(image_path, title, output_path):
    img = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    band_h = 105
    draw.rectangle([0, 0, WIDTH, band_h], fill="white")
    f = font_for(44)
    while True:
        box = draw.textbbox((0, 0), title, font=f)
        if box[2] - box[0] <= WIDTH - 80 or getattr(f, "size", 44) <= 24:
            break
        f = font_for(max(24, getattr(f, "size", 44) - 2))
    tw = box[2] - box[0]
    draw.text(((WIDTH - tw) / 2, 25), title, fill="black", font=f)
    img.save(output_path, quality=95)

# -----------------------------
# Bộ máy Bàn tay & Quỹ đạo nét vẽ (Khôi phục Nearest-Neighbor Sorting)
# -----------------------------
def load_hand_asset(hand_path, target_width=320):
    p = Path(hand_path) if hand_path and Path(hand_path).exists() else Path("hand.png")
    if p.exists():
        pil_hand = Image.open(p).convert("RGBA")
    else:
        pil_hand = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
    w, h = pil_hand.size
    new_h = int(h * (target_width / w))
    pil_hand = pil_hand.resize((target_width, new_h), Image.Resampling.LANCZOS)
    hand_np = np.array(pil_hand)
    bgr = cv2.cvtColor(hand_np[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = hand_np[:, :, 3]
    ys, xs = np.where(alpha > 120)
    tip_x, tip_y = (int(xs[np.argmin(xs + ys * 1.15)]), int(ys[np.argmin(xs + ys * 1.15)])) if len(xs) > 0 else (0, 0)
    return bgr, alpha, tip_x, tip_y

def sort_contours_nn(contours, start_pt=(100, 150)):
    """Thuật toán tối ưu đường đi nét gần nhất để tay không bị nhảy cóc"""
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

def extract_drawing_path(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return [(WIDTH // 2, HEIGHT // 2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    # Chia 4 phân khu: Tiêu đề -> Trái -> Giữa -> Phải
    title_mask = np.zeros_like(binary)
    title_mask[:105, :] = binary[:105, :]

    left_mask = np.zeros_like(binary)
    left_mask[105:, :int(WIDTH * 0.38)] = binary[105:, :int(WIDTH * 0.38)]

    center_mask = np.zeros_like(binary)
    center_mask[105:, int(WIDTH * 0.38):int(WIDTH * 0.68)] = binary[105:, int(WIDTH * 0.38):int(WIDTH * 0.68)]

    right_mask = np.zeros_like(binary)
    right_mask[105:, int(WIDTH * 0.68):] = binary[105:, int(WIDTH * 0.68):]

    t_cnts, _ = cv2.findContours(title_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    l_cnts, _ = cv2.findContours(left_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c_cnts, _ = cv2.findContours(center_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    r_cnts, _ = cv2.findContours(right_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Sắp xếp tiêu đề từ trái qua phải
    t_sorted = sorted(t_cnts, key=lambda c: cv2.boundingRect(c)[0])

    # Áp dụng thuật toán tìm nét gần nhất cho từng phân khu
    l_sorted = sort_contours_nn(l_cnts, start_pt=(150, 200))
    c_sorted = sort_contours_nn(c_cnts, start_pt=(int(WIDTH * 0.5), 200))
    r_sorted = sort_contours_nn(r_cnts, start_pt=(int(WIDTH * 0.8), 200))

    all_contours = t_sorted + l_sorted + c_sorted + r_sorted

    trajectory = []
    for c in all_contours:
        pts = c.reshape(-1, 2)
        for p in pts[::3]:
            trajectory.append((int(p[0]), int(p[1])))

    if len(trajectory) < 40:
        trajectory = [(x, y) for y in range(120, 680, 45) for x in range(80, 1200, 25)]
    return trajectory

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
    roi = frame_bgr[y1:y2, x1:x2]
    frame_bgr[y1:y2, x1:x2] = (sub_hand * sub_alpha + roi * (1.0 - sub_alpha)).astype(np.uint8)

def ease_in_out(t):
    return 0.5 * (1.0 - math.cos(math.pi * t))

# -----------------------------
# Chế độ 1: Độc Bản (Tay vẽ + Camera lướt theo nét + Zoom out)
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

    trajectory = extract_drawing_path(image_path)
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

        # Cập nhật nét vẽ và vị trí tay (Khôi phục micro-jitter rung nhẹ tự nhiên)
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

        # Tính toán camera lướt theo ngòi bút và zoom out kết màn
        if f_idx < draw_frames:
            scale = 1.25
            smooth_cx = smooth_cx * 0.95 + hand_pos_x * 0.05
            smooth_cy = smooth_cy * 0.95 + hand_pos_y * 0.05
        else:
            out_prog = ease_in_out((f_idx - draw_frames) / max(1, total_frames - draw_frames))
            scale = 1.25 - 0.25 * out_prog
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
        raise RuntimeError("FFmpeg thất bại trong chế độ Độc bản.")

# -----------------------------
# Chế độ 2: Kiến Thức Thú Vị (Chỉ lia máy + Zoom động, ẩn tay)
# -----------------------------
def render_scene_kttv(image_path, duration, output_path):
    total_frames = max(1, round(duration * FPS))
    original_bgr = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output_path)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    pt_left = (WIDTH * 0.32, HEIGHT * 0.58)
    pt_right = (WIDTH * 0.68, HEIGHT * 0.58)
    pt_center = (WIDTH * 0.50, HEIGHT * 0.50)

    for f_idx in range(total_frames):
        p = f_idx / max(1, total_frames - 1)
        if p < 0.35:
            scale = 1.35 + 0.05 * (p / 0.35)
            cx, cy = pt_left
        elif p < 0.60:
            sub_p = ease_in_out((p - 0.35) / 0.25)
            scale = 1.40
            cx = pt_left[0] + (pt_right[0] - pt_left[0]) * sub_p
            cy = pt_left[1] + (pt_right[1] - pt_left[1]) * sub_p
        elif p < 0.85:
            sub_p = ease_in_out((p - 0.60) / 0.25)
            scale = 1.40 - 0.40 * sub_p
            cx = pt_right[0] + (pt_center[0] - pt_right[0]) * sub_p
            cy = pt_right[1] + (pt_center[1] - pt_right[1]) * sub_p
        else:
            scale = 1.0
            cx, cy = pt_center

        crop_w, crop_h = int(WIDTH / scale), int(HEIGHT / scale)
        x1 = max(0, min(WIDTH - crop_w, int(cx - crop_w / 2)))
        y1 = max(0, min(HEIGHT - crop_h, int(cy - crop_h / 2)))
        crop = original_bgr[y1:y1 + crop_h, x1:x1 + crop_w]
        frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

# -----------------------------
# Chế độ 3: Bảng trắng cổ điển (Góc máy tĩnh, tay vẽ)
# -----------------------------
def render_scene_classic_hand(image_path, duration, output_path, hand_path):
    total_frames = max(1, round(duration * FPS))
    draw_frames = int(max(2.0, min(duration - 1.0, duration * 0.75)) * FPS)
    retract_frames = int(0.5 * FPS)
    original_bgr = cv2.resize(cv2.imread(str(image_path)), (WIDTH, HEIGHT))
    white_canvas = np.full_like(original_bgr, 255)
    reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    trajectory = extract_drawing_path(image_path)
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
            for pt in trajectory[prev_idx:curr_idx]:
                cv2.circle(reveal_mask, pt, 24, 255, -1)
            target_pt = trajectory[min(curr_idx, len(trajectory) - 1)] if trajectory else (WIDTH // 2, HEIGHT // 2)
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
def render_batch(batch_audio, scenes, batch_dir, hand_path, style, progress_callback=None):
    scene_videos = []
    total = len(scenes)
    for i, s in enumerate(scenes, 1):
        img_raw = batch_dir / f"scene_{i:03d}_raw.png"
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"
        if not img.exists():
            cloudflare_image(s["visual_prompt"], cloudflare_account_id, cloudflare_token, image_model, img_raw, image_timeout)
            add_title(img_raw, s["title"], img)
        duration = max(1.0, float(s["end"]) - float(s["start"]))

        # Phân luồng chính xác theo đúng lựa chọn trên Sidebar
        if "Độc bản" in style:
            render_scene_hybrid(img, duration, vid, hand_path)
        elif "Kiến Thức Thú Vị" in style:
            render_scene_kttv(img, duration, vid)
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
if st.sidebar.button("🔎 KIỂM TRA ẢNH BỐ CỤC 3 PHẦN", use_container_width=True):
    try:
        with st.spinner("Cloudflare đang vẽ tranh 3 phân khu..."):
            test_img = test_cloudflare_api(cloudflare_account_id, cloudflare_token, image_model, 120)
        st.success("✅ Ảnh tạo thành công — Bố cục rộng, nhân vật bán thân, sạch chữ 100%!")
        st.image(test_img, caption="Ảnh mẫu bố cục 3 vùng (Left - Center - Right)", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")

audio = st.file_uploader("🎤 Tải lên tệp ghi âm giọng nói", type=["mp3", "m4a", "wav", "ogg", "webm", "mp4"])

if audio:
    st.audio(audio)

    if st.button("🚀 BẮT ĐẦU TẠO VIDEO", type="primary", use_container_width=True):
        if not groq_key or not cloudflare_account_id or not cloudflare_token:
            st.error("Vui lòng nhập đầy đủ Groq API Key, Cloudflare Account ID và Token.")
            st.stop()

        root = Path(tempfile.mkdtemp(prefix="wb_ai_"))
        try:
            source = root / audio.name
            source.write_bytes(audio.getbuffer())
            duration = ffprobe_duration(source)
            st.info(f"Thời lượng âm thanh: {duration/60:.2f} phút. Hệ thống xử lý theo từng đợt 5 phút chuẩn.")

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

                status.write(f"✂️ Đợt {idx+1}/{len(valid_chunks)} — Lên kịch bản 3 phân khu bao quát...")
                scenes = make_scene_plan(client, batch_text, bstart, bdur, planner_model, scene_min, scene_max, max_scenes_per_batch)

                st.write(f"**Đợt {idx+1}: {bdur:.1f}s → {len(scenes)} cảnh**")
                for si, s in enumerate(scenes, 1):
                    st.caption(f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}")

                batch_work = root / f"work_{idx+1:03d}"
                batch_work.mkdir()

                status.write(f"🎨 Đợt {idx+1}/{len(valid_chunks)} — Đang vẽ tranh và render hiệu ứng...")
                def cb(frac, idx=idx):
                    progress.progress(min(1.0, (idx + frac) / len(valid_chunks)))

                bv = render_batch(chunk, scenes, batch_work, hand_path, draw_style, cb)
                saved_batch = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bv, saved_batch)
                batch_videos.append(saved_batch)
                all_scene_count += len(scenes)
                shutil.rmtree(batch_work, ignore_errors=True)

            progress.progress(1.0)
            status.write("🎬 Đang kết hợp video hoàn chỉnh...")
            final = root / "video_hoan_chinh.mp4"
            concat_batches(batch_videos, final)

            st.success(f"Hoàn thành xuất sắc! Đã tạo {all_scene_count} cảnh chuẩn nét vẽ tay và đồng bộ 100% âm thanh.")
            st.video(str(final))
            st.download_button(
                "⬇️ TẢI VIDEO MP4 VỀ MÁY",
                data=final.read_bytes(),
                file_name="video_hoan_chinh.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)
        finally:
            pass
