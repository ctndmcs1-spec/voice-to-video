import os
import re
import io
import json
import math
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
import cv2
import numpy as np

APP_TITLE = "Xưởng Video Vẽ Tay AI"
BATCH_SECONDS = 5 * 60  # 5 phút chuẩn
FPS = 30
WIDTH = 1280
HEIGHT = 720
HF_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"

# -----------------------------
# Giao diện / Cấu hình
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="✏️", layout="wide")

st.title("✏️ Xưởng Tạo Video Vẽ Bảng Trắng AI")
st.caption("Giọng nói → Phân cảnh Groq → Vẽ tranh FLUX (Hugging Face) → Nét bút vẽ tay OpenCV → MP4")

with st.sidebar:
    st.header("🔑 Cấu hình API")
    groq_key = st.text_input(
        "Khóa Groq API (Groq API Key)",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password",
        help="Dùng cho nhận diện giọng nói và lên kịch bản phân cảnh.",
    )
    hf_token = st.text_input(
        "Khóa Hugging Face Token",
        value=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "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")),
        type="password",
        help="Dùng để tạo tranh FLUX.1-schnell miễn phí.",
    )

    st.header("🧠 Mô hình Groq")
    stt_model = st.selectbox(
        "Mô hình nghe giọng nói (Voice → Text)",
        ["whisper-large-v3", "whisper-large-v3-turbo"],
        index=0,
    )
    planner_model = st.selectbox(
        "Mô hình biên kịch kịch bản",
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "groq/compound-mini"],
        index=0,
    )

    st.header("🎬 Cài đặt hiệu ứng")
    scene_min = st.slider("Thời lượng cảnh tối thiểu (giây)", 15, 25, 15)
    scene_max = st.slider("Thời lượng cảnh tối đa (giây)", 20, 30, 30)
    if scene_max < scene_min:
        scene_max = scene_min

    draw_style = st.selectbox(
        "Phong cách vẽ hoạt họa",
        [
            "Bảng trắng + Bàn tay đưa nét vẽ",
            "Bảng trắng + Bàn tay vẽ + Phóng to nhẹ",
            "Chuyển động đồ họa sạch (Không hiện tay)",
        ],
    )

    st.header("⚙️ Giới hạn an toàn")
    max_scenes_per_batch = st.slider("Số cảnh tối đa mỗi đợt 5 phút", 5, 25, 20)
    image_timeout = st.slider("Thời gian chờ tạo ảnh (giây)", 30, 180, 90)

# -----------------------------
# Tiện ích hệ thống
# -----------------------------
def run_cmd(cmd, timeout=600):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
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
    raise ValueError("AI không trả về cấu trúc JSON hợp lệ")

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

def make_scene_plan(client, transcript_text, batch_start, batch_duration, model, min_s, max_s, max_scenes):
    system = f"""
You are the visual director for a Vietnamese whiteboard explainer channel.

Task:
Turn a voice transcript into coherent visual scenes.

Hard rules:
1. Each scene must be {min_s}-{max_s} seconds.
2. Do NOT cut in the middle of an important idea if a nearby boundary works better.
3. Each scene gets EXACTLY ONE main infographic image.
4. One image must visually summarize ALL important ideas spoken in that scene.
5. The image is a whiteboard educational infographic: white background, black hand-drawn ink, simple expressive characters, arrows, objects, diagrams, a few restrained accent colors.
6. Do not put Vietnamese words or tiny labels inside the generated image. The Python tool will add the accurate title itself.
7. Do not make generic filler images. Every object must be justified by the voice.
8. Avoid copyrighted characters, logos and real-person likenesses.
9. The visual prompt must be in English.
10. Safety: NEVER use words like blood, kill, suicide, weapon. Represent dark themes symbolically (e.g., storm clouds, broken chains, stressed figure holding head, scales of justice, warning signposts).
11. Return ONLY valid JSON.

JSON:
{{
  "scenes": [
    {{
      "start": 0.0,
      "end": 20.0,
      "title": "short Vietnamese title",
      "summary": "one sentence in Vietnamese",
      "visual_prompt": "detailed symbolic whiteboard illustration in English"
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
            "title": "Tổng kết",
            "summary": (transcript_text[:120] if transcript_text else "Kết thúc nội dung"),
            "visual_prompt": "A symbolic whiteboard educational drawing of a person thinking, question marks, abstract diagrams, clean white background, minimalist black ink art",
        }]

    clean[0]["start"] = 0.0

    # Lấp khoảng trống giữa các cảnh
    for i in range(len(clean) - 1):
        clean[i]["end"] = clean[i + 1]["start"]

    clean[-1]["end"] = batch_duration
    return clean

# -----------------------------
# Hugging Face FLUX.1 Engine
# -----------------------------
def hf_flux_request(prompt, token, timeout=90):
    token = (token or "").strip()
    safe_prompt = sanitize_prompt_text(prompt)
    full_prompt = (
        f"{safe_prompt}, single coherent whiteboard infographic, 16:9 landscape composition, "
        "pure white paper background, hand-drawn black ink line art, simple expressive educational illustration, "
        "clean composition, subtle red and blue accent strokes only, clear visual hierarchy, arrows, "
        "empty band near top for title, NO words, NO letters, NO text, NO watermark, 2D vector style"
    )

    # 1. Gọi trực tiếp qua Router mới của Hugging Face
    endpoints = [
        f"https://router.huggingface.co/hf-inference/models/{HF_FLUX_MODEL}",
        f"https://router.huggingface.co/models/{HF_FLUX_MODEL}",
    ]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/jpeg",
    }
    payload = {"inputs": full_prompt}

    last_err = None
    for api_url in endpoints:
        for attempt in range(1, 4):
            try:
                r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
                if r.status_code == 200 and len(r.content) > 1000:
                    return r.content
                if r.status_code == 503:
                    time.sleep(8 * attempt)
                    continue
                if r.status_code == 401:
                    raise RuntimeError("Hugging Face lỗi 401: Token không đúng hoặc chưa có quyền Read.")
                if r.status_code >= 400:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)
                time.sleep(2 * attempt)

    # 2. Cơ chế tự cứu thông minh (Fallback): Tự chuyển sang FLUX trực tiếp nếu HF bị lỗi mạng/DNS
    try:
        fb_url = f"https://image.pollinations.ai/prompt/{quote(full_prompt, safe='')}"
        r_fb = requests.get(fb_url, params={"model": "flux", "width": WIDTH, "height": HEIGHT, "nologo": "true"}, timeout=60)
        if r_fb.status_code == 200 and len(r_fb.content) > 1000:
            return r_fb.content
    except Exception:
        pass

    raise RuntimeError(f"Không thể kết nối máy chủ tạo ảnh: {last_err}")

def generate_image(prompt, token, output_path, timeout):
    data = hf_flux_request(prompt, token, timeout)
    Path(output_path).write_bytes(data)
    try:
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            im.convert("RGB").resize((WIDTH, HEIGHT)).save(output_path, quality=94)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"File ảnh tạo ra không hợp lệ: {e}")

def test_hf_api(token, timeout):
    data = hf_flux_request(
        "A simple black ink whiteboard drawing of an idea light bulb and an open book, minimal composition, white background",
        token,
        timeout,
    )
    return Image.open(io.BytesIO(data)).convert("RGB")

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
# Bộ máy vẽ tay từng nét (OpenCV)
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

    ys, xs = np.where(alpha > 120)
    if len(xs) > 0:
        score = xs + ys * 1.15
        tip_idx = np.argmin(score)
        tip_x = int(xs[tip_idx])
        tip_y = int(ys[tip_idx])
    else:
        tip_x, tip_y = 0, 0
    return bgr, alpha, tip_x, tip_y

def extract_drawing_path(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return [(WIDTH // 2, HEIGHT // 2)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    title_mask = np.zeros_like(binary)
    title_mask[:105, :] = binary[:105, :]

    body_mask = np.zeros_like(binary)
    body_mask[105:, :] = binary[105:, :]

    title_contours, _ = cv2.findContours(title_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    body_contours, _ = cv2.findContours(body_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    title_sorted = sorted(title_contours, key=lambda c: cv2.boundingRect(c)[0])

    def c_center(c):
        M = cv2.moments(c)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h // 2)

    valid_body = [c for c in body_contours if cv2.arcLength(c, False) > 12]
    body_sorted = []
    if valid_body:
        curr = (100, 150)
        rem = valid_body[:]
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
            body_sorted.append(chosen)
            curr = c_center(chosen)

    all_contours = title_sorted + body_sorted
    trajectory = []
    for c in all_contours:
        pts = c.reshape(-1, 2)
        sampled = pts[::3]
        for p in sampled:
            trajectory.append((int(p[0]), int(p[1])))

    if len(trajectory) < 40:
        trajectory = []
        for y in range(120, 680, 45):
            xs = range(80, 1200, 25) if (y // 45) % 2 == 0 else range(1200, 80, -25)
            for x in xs:
                trajectory.append((x, y))
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

def render_scene(image_path, duration, output_path, hand_path, style):
    total_frames = max(1, round(duration * FPS))
    draw_duration = max(2.0, min(duration - 1.0, duration * 0.75))
    draw_frames = int(draw_duration * FPS)
    retract_frames = int(0.5 * FPS)

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise RuntimeError(f"Không đọc được file ảnh: {image_path}")
    original_bgr = cv2.resize(original_bgr, (WIDTH, HEIGHT))
    white_canvas = np.full_like(original_bgr, 255)
    reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    trajectory = extract_drawing_path(image_path)
    hand_bgr, hand_alpha, tip_x, tip_y = load_hand_asset(hand_path, target_width=330)

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

    last_tip = trajectory[-1] if trajectory else (WIDTH // 2, HEIGHT // 2)

    for f_idx in range(total_frames):
        hand_visible = False
        hand_pos_x, hand_pos_y = 0, 0

        if f_idx < draw_frames:
            curr_idx = int((f_idx + 1) / draw_frames * len(trajectory))
            prev_idx = int(f_idx / draw_frames * len(trajectory))
            step_pts = trajectory[prev_idx:curr_idx]

            for pt in step_pts:
                cv2.circle(reveal_mask, pt, 22, 255, -1)

            if step_pts:
                target_pt = step_pts[-1]
            else:
                target_pt = trajectory[min(curr_idx, len(trajectory) - 1)]

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

        if hand_visible and ("Không hiện tay" not in style and "Clean" not in style):
            paste_hand(frame, hand_bgr, hand_alpha, hand_pos_x - tip_x, hand_pos_y - tip_y)

        if "Phóng to" in style or "zoom" in style.lower():
            scale = 1.0 + 0.05 * (f_idx / total_frames)
            cw, ch = int(WIDTH / scale), int(HEIGHT / scale)
            x1 = (WIDTH - cw) // 2
            y1 = (HEIGHT - ch) // 2
            crop = frame[y1:y1 + ch, x1:x1 + cw]
            frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg thất bại: {err[-2000:]}")

# -----------------------------
# Quy trình render theo đợt (Batch)
# -----------------------------
def render_batch(batch_audio, scenes, batch_dir, hand_path, style, progress_callback=None):
    scene_videos = []
    total = len(scenes)
    for i, s in enumerate(scenes, 1):
        img_raw = batch_dir / f"scene_{i:03d}_raw.png"
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"
        if not img.exists():
            generate_image(s["visual_prompt"], hf_token, img_raw, image_timeout)
            add_title(img_raw, s["title"], img)
        duration = max(1.0, float(s["end"]) - float(s["start"]))
        render_scene(img, duration, vid, hand_path, style)
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
if st.sidebar.button("🔎 KIỂM TRA TẠO ẢNH HUGGING FACE", use_container_width=True):
    try:
        with st.spinner("Đang kết nối tới mô hình FLUX.1-schnell..."):
            test_img = test_hf_api(hf_token, 60)
        st.success("Hugging Face FLUX kết nối rất tốt — Sẵn sàng tạo ảnh!")
        st.image(test_img, caption="Ảnh minh họa thử nghiệm từ FLUX.1-schnell", use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi kiểm tra: {e}")

audio = st.file_uploader(
    "🎤 Tải lên tệp ghi âm giọng nói",
    type=["mp3", "m4a", "wav", "ogg", "webm", "mp4", "mpeg", "mpga"],
    help="Hỗ trợ các định dạng âm thanh phổ biến.",
)

if audio:
    st.audio(audio)

    if st.button("🚀 BẮT ĐẦU TẠO VIDEO", type="primary", use_container_width=True):
        if not groq_key:
            st.error("Vui lòng nhập Khóa Groq API.")
            st.stop()
        if not hf_token:
            st.error("Vui lòng nhập Khóa Hugging Face Token.")
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

            valid_chunks = []
            for bi, chunk in enumerate(chunks):
                cdur = ffprobe_duration(chunk)
                if cdur < 5.0 and bi > 0:
                    continue
                valid_chunks.append((bi, chunk, cdur))

            for idx, (bi, chunk, bdur) in enumerate(valid_chunks):
                bstart = bi * BATCH_SECONDS
                status.write(f"🧠 Đợt {idx+1}/{len(valid_chunks)} — Đang nhận diện giọng nói...")
                tr = transcribe_file(client, chunk, stt_model)
                segs = normalize_segments(tr, bstart)
                batch_text = "\n".join(
                    f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}"
                    for x in segs
                )

                status.write(f"✂️ Đợt {idx+1}/{len(valid_chunks)} — Đang lập kịch bản phân cảnh...")
                scenes = make_scene_plan(
                    client,
                    batch_text,
                    bstart,
                    bdur,
                    planner_model,
                    scene_min,
                    scene_max,
                    max_scenes_per_batch,
                )

                st.write(f"**Đợt {idx+1}: {bdur:.1f}s → {len(scenes)} cảnh**")
                for si, s in enumerate(scenes, 1):
                    st.caption(
                        f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}"
                    )

                batch_work = root / f"work_{idx+1:03d}"
                batch_work.mkdir()

                status.write(f"🎨 Đợt {idx+1}/{len(valid_chunks)} — Đang tạo tranh FLUX và render hiệu ứng nét vẽ...")
                def cb(frac, idx=idx):
                    progress.progress(min(1.0, (idx + frac) / len(valid_chunks)))

                bv = render_batch(
                    chunk, scenes, batch_work, hand_path, draw_style, cb
                )

                saved_batch = root / f"batch_final_{idx+1:03d}.mp4"
                shutil.copy2(bv, saved_batch)
                batch_videos.append(saved_batch)
                all_scene_count += len(scenes)

                shutil.rmtree(batch_work, ignore_errors=True)

            progress.progress(1.0)
            status.write("🎬 Đang kết hợp video hoàn chỉnh...")

            final = root / "whiteboard_final.mp4"
            concat_batches(batch_videos, final)

            st.success(
                f"Đã tạo thành công {all_scene_count} cảnh chuẩn nét vẽ tay và đồng bộ 100% âm thanh!"
            )
            st.video(str(final))
            st.download_button(
                "⬇️ TẢI VIDEO MP4 VỀ MÁY",
                data=final.read_bytes(),
                file_name="video_ve_tay_hoan_chinh.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)
            st.warning("Nếu gặp lỗi, vui lòng kiểm tra thông báo trong mục Quản lý ứng dụng.")
        finally:
            pass
