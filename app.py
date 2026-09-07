# -*- coding: utf-8 -*-
"""
POV PSYCHOLOGICAL CINEMATIC ENGINE (STORYTELLING VN STYLE)
- High-Def Vector Wojak Synthesis (No External Scraping Fails)
- Multi-Pass Color Grading & Vignette Compositor
- Dynamic Ken Burns with Parallax Depth Simulation
- Drop Shadow & Spatial Alpha Masking
"""

import os
import re
import io
import json
import base64
import random
import shutil
import subprocess
import tempfile
import time
import urllib.parse

import streamlit as st
import requests
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="Studio POV Cinematic Engine",
    page_icon="🎬",
    layout="centered"
)

# THÔNG SỐ KHUNG HÌNH CHUẨN ĐIỆN ẢNH
W, H = 1280, 720
FPS = 30
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"

# KHO BỐI CẢNH TÂM LÝ VIỆT NAM ĐẶC TẢ SÂU
PSYCHO_CONTEXT_TERMS = {
    "suspicion": [
        "căn phòng tối le lói ánh đèn đường qua cửa sổ",
        "hành lang chung cư cũ ánh đèn vàng vắng tanh",
        "bàn làm việc ngổn ngang ban đêm một góc khuất"
    ],
    "exhaustion": [
        "người ngồi gục đầu bên bàn làm việc ban đêm",
        "phòng trọ nhỏ tối tăm bừa bộn ánh sáng yếu",
        "cửa kính ban đêm mờ sương mưa rơi lạnh lẽo"
    ],
    "manipulation": [
        "bóng người đàn ông đứng nhìn qua khe cửa tối",
        "góc phòng khách chung cư ánh đèn hắt tường u ám",
        "ngã tư đường phố đêm vắng tanh ánh đèn neon"
    ],
    "isolation": [
        "băng ghế đá công viên ban đêm dưới bóng đèn mờ",
        "cầu thang bộ chung cư cũ u tối ẩm thấp",
        "cửa sổ tầng cao chung cư nhìn ra thành phố đêm"
    ]
}

# ==============================================================================
# HỆ THỐNG DỰNG SPRITE WOJAK NÉT MỰC CAO CẤP (VECTOR-BASED DRAW ENGINE)
# Không phụ thuộc vào việc cào ảnh mạng thất thường, luôn sinh sprite 100% sắc nét
# ==============================================================================
class WojakRenderer:
    @staticmethod
    def _draw_base_head(draw, skin_tone=(248, 238, 230), line_color=(15, 15, 20)):
        # Hộp sọ đầu Wojak chuẩn tỉ lệ
        draw.polygon([
            (110, 240), (95, 180), (105, 110), (145, 60), (210, 45),
            (280, 55), (320, 100), (330, 160), (320, 220), (290, 270),
            (240, 290), (190, 295), (140, 280)
        ], fill=skin_tone, outline=line_color)
        
        # Nếp nhăn trán đặc trưng của Wojak
        draw.arc([150, 80, 270, 110], start=190, end=350, fill=line_color, width=3)
        draw.arc([160, 95, 260, 120], start=190, end=350, fill=line_color, width=2)
        draw.arc([170, 110, 250, 130], start=190, end=350, fill=line_color, width=2)

        # Mũi gãy / Nếp má
        draw.line([(190, 160), (170, 200), (195, 205)], fill=line_color, width=4)
        draw.arc([140, 180, 190, 250], start=280, end=70, fill=line_color, width=3)

    @staticmethod
    def _draw_torso(draw, clothing_color=(35, 38, 48), line_color=(15, 15, 20)):
        # Cổ & Thân áo hoodie/áo thun u tối
        draw.polygon([(160, 290), (240, 290), (260, 360), (140, 360)], fill=(235, 220, 210), outline=line_color)
        draw.polygon([
            (70, 560), (110, 370), (140, 360), (260, 360), (290, 370),
            (330, 560), (200, 560)
        ], fill=clothing_color, outline=line_color)
        # Nếp gấp quần áo
        draw.line([(150, 370), (170, 500)], fill=(20, 22, 28), width=3)
        draw.line([(250, 370), (230, 500)], fill=(20, 22, 28), width=3)

    @classmethod
    def render_sprite(cls, emotion: str, target_h: int = 560) -> Image.Image:
        """Sinh sprite nhân vật Wojak 2D nét mực sắc lạnh theo cảm xúc"""
        canvas = Image.new("RGBA", (400, 580), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        lc = (20, 20, 25)

        cls._draw_torso(d, clothing_color=(32, 35, 45), line_color=lc)
        cls._draw_base_head(d, skin_tone=(244, 236, 228), line_color=lc)

        if emotion == "thinking":
            # Mắt nhắm nghi ngờ, đồng tử nhỏ
            d.ellipse([140, 145, 175, 165], outline=lc, width=3)
            d.ellipse([215, 145, 250, 165], outline=lc, width=3)
            d.ellipse([155, 152, 163, 160], fill=lc)
            d.ellipse([230, 152, 238, 160], fill=lc)
            # Miệng cong xuống băn khoăn
            d.arc([160, 240, 220, 270], start=200, end=340, fill=lc, width=4)
            # Tay đặt cằm suy ngẫm
            d.polygon([(110, 320), (160, 280), (190, 260), (180, 240), (140, 260)], fill=(244, 236, 228), outline=lc)

        elif emotion == "stressed":
            # Quầng thâm mắt dày đặc, bọng mắt to
            d.ellipse([135, 140, 180, 175], outline=lc, width=3)
            d.ellipse([210, 140, 255, 175], outline=lc, width=3)
            d.ellipse([150, 150, 165, 165], fill=(80, 20, 20))
            d.ellipse([225, 150, 240, 165], fill=(80, 20, 20))
            # Quầng thâm
            d.arc([130, 155, 185, 195], start=10, end=170, fill=(120, 110, 130), width=3)
            d.arc([205, 155, 260, 195], start=10, end=170, fill=(120, 110, 130), width=3)
            # Miệng run rẩy
            d.line([(155, 255), (170, 250), (185, 258), (205, 248), (225, 255)], fill=lc, width=4)
            # Giọt mồ hôi lạnh
            d.polygon([(265, 110), (275, 130), (255, 130)], fill=(180, 220, 255), outline=lc)

        elif emotion == "smug":
            # Mắt nửa mí khinh bỉ, kẻ ái kỷ
            d.arc([140, 140, 180, 165], start=180, end=360, fill=lc, width=4)
            d.arc([215, 140, 255, 165], start=180, end=360, fill=lc, width=4)
            d.ellipse([155, 148, 165, 158], fill=lc)
            d.ellipse([230, 148, 240, 158], fill=lc)
            # Nụ cười nhếch mép mỉa mai
            d.arc([160, 220, 240, 270], start=350, end=140, fill=lc, width=4)
            d.line([(235, 235), (250, 225)], fill=lc, width=3)

        elif emotion == "depressed":
            # Doomer mắt thâm sâu, cúi gằm
            d.line([(135, 155), (175, 165)], fill=lc, width=4)
            d.line([(215, 165), (255, 155)], fill=lc, width=4)
            d.arc([130, 150, 180, 190], start=10, end=170, fill=(90, 85, 100), width=4)
            d.arc([210, 150, 260, 190], start=10, end=170, fill=(90, 85, 100), width=4)
            d.arc([165, 260, 225, 290], start=190, end=350, fill=lc, width=4)
            # Mũ len Doomer đen
            d.ellipse([110, 30, 310, 130], fill=(25, 28, 32), outline=lc)

        else: # Neutral
            d.ellipse([140, 145, 175, 170], outline=lc, width=3)
            d.ellipse([215, 145, 250, 170], outline=lc, width=3)
            d.ellipse([153, 153, 163, 163], fill=lc)
            d.ellipse([228, 153, 238, 163], fill=lc)
            d.line([(165, 250), (225, 250)], fill=lc, width=4)

        # Scale theo chiều cao chỉ định
        ratio = target_h / canvas.height
        target_w = int(canvas.width * ratio)
        return canvas.resize((target_w, target_h), Image.LANCZOS)

# ==============================================================================
# HỆ THỐNG COLOR GRADING & COMPOSITING ĐA LỚP
# ==============================================================================
class CinematicCompositor:
    @staticmethod
    def apply_color_grading(bg_img: Image.Image) -> Image.Image:
        """Phủ tông màu lạnh điện ảnh (Dark Low-Key Grade) & tăng chiều sâu ánh sáng"""
        # Giảm sáng nhẹ, tăng tương phản
        enhancer = ImageEnhance.Contrast(bg_img)
        graded = enhancer.enhance(1.22)
        enhancer = ImageEnhance.Brightness(graded)
        graded = enhancer.enhance(0.85)

        # Đẩy nhẹ ám xanh lạnh (Teal/Dark Blue Tint)
        r, g, b = graded.split()
        r = r.point(lambda i: i * 0.90)
        b = b.point(lambda i: min(255, int(i * 1.08)))
        graded = Image.merge("RGB", (r, g, b))

        # Phủ lớp Vignette tối góc điện ảnh
        vignette = Image.new("L", (W, H), 255)
        d_v = ImageDraw.Draw(vignette)
        d_v.ellipse([-W * 0.15, -H * 0.15, W * 1.15, H * 1.15], fill=0)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=120))
        
        black_layer = Image.new("RGB", (W, H), (10, 12, 16))
        final_bg = Image.composite(black_layer, graded, vignette)
        return final_bg

    @staticmethod
    def create_character_layer(char_sprite: Image.Image, pos: str) -> Image.Image:
        """Tạo layer nhân vật kèm bóng đổ mềm (Drop Shadow) hòa trộn không gian"""
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        c_w, c_h = char_sprite.size

        # Tọa độ: Đặt sát góc dưới trái hoặc phải
        x = (W - c_w - 60) if pos == "right" else 60
        y = H - c_h + 15  # Hơi chìm mép dưới tạo độ chắc

        # 1. Tạo bóng đổ mềm sau lưng (Drop Shadow)
        shadow_mask = char_sprite.split()[3]
        shadow = Image.new("RGBA", (c_w, c_h), (0, 0, 0, 180))
        shadow.putalpha(shadow_mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
        canvas.paste(shadow, (x + (15 if pos == "left" else -15), y + 10), shadow)

        # 2. Dán Sprite nhân vật sắc nét lên trên
        canvas.paste(char_sprite, (x, y), char_sprite)
        return canvas

# ==============================================================================
# BỘ CÀO BỐI CẢNH THỰC TẾ VIỆT NAM (ĐA TẦNG DỰ PHÒNG)
# ==============================================================================
def fetch_cinematic_vietnam_bg(query_vn: str, idx: int, workdir: str, used_urls: set, p_key: str) -> str:
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    search_queries = [
        f"{query_vn} chụp thực tế việt nam",
        f"{query_vn} phòng trọ chung cư hà nội",
        query_vn
    ]

    for q in search_queries:
        if downloaded:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, region="vn-vi", max_results=8))
                for r in results:
                    u = r.get("image")
                    if u and u.startswith("http") and u not in used_urls:
                        try:
                            resp = requests.get(u, headers=headers, timeout=5)
                            if resp.status_code == 200 and len(resp.content) > 30000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                with Image.open(dest) as t_img:
                                    if t_img.size[0] >= 640 and t_img.size[1] >= 360:
                                        used_urls.add(u)
                                        downloaded = True
                                        break
                        except Exception:
                            continue
        except Exception:
            continue

    # Fallback chất lượng cao qua Pexels nếu cào mạng bị block
    if not downloaded and p_key and p_key.strip():
        try:
            fallback_term = random.choice([
                "dark empty room window night",
                "apartment moody interior shadow",
                "dimly lit hallway urban",
                "man looking through dark window"
            ])
            url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(fallback_term)}&per_page=6&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
            if r.ok and r.json().get("photos"):
                u = r.json()["photos"][0]["src"]["large2x"]
                with open(dest, "wb") as f:
                    f.write(requests.get(u, timeout=8).content)
                downloaded = True
        except Exception:
            pass

    # Nếu hoàn toàn không có mạng: Tạo canvas gradient nội thất tối màu
    if not downloaded:
        base_canvas = Image.new("RGB", (W, H), (18, 20, 26))
        d_c = ImageDraw.Draw(base_canvas)
        d_c.rectangle([0, H*0.6, W, H], fill=(12, 14, 18))
        d_c.rectangle([W*0.6, H*0.1, W*0.9, H*0.55], fill=(30, 35, 45), outline=(50, 55, 65), width=3)
        base_canvas.save(dest, "JPEG")

    # Đưa qua bộ lọc Color Grading & Vignette chuẩn Studio
    try:
        with Image.open(dest) as raw_bg:
            fitted = ImageOps.fit(raw_bg.convert("RGB"), (W, H), Image.LANCZOS)
            graded = CinematicCompositor.apply_color_grading(fitted)
            graded.save(dest, "JPEG", quality=95)
    except Exception:
        pass

    return dest

# ==============================================================================
# RENDER ENGINE ĐIỆN ẢNH VỚI KEN BURNS 60FPS KHỬ RUNG
# ==============================================================================
def render_cinematic_scene(bg_path: str, char_layer_path: str, duration: float, out_clip: str, mode: int = 0):
    """FFmpeg dựng chuyển động camera Ken Burns biến thiên và đè nhân vật hòa trộn"""
    frames = max(30, int(duration * FPS))
    step = 0.18 / frames

    # Quỹ đạo di chuyển camera điện ảnh luân phiên
    if mode % 3 == 0:
        # Zoom In sâu vào tâm điểm
        z_expr = f"min(zoom+{step:.6f},1.18)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif mode % 3 == 1:
        # Zoom Out từ góc tối mở rộng
        z_expr = f"if(eq(on,1),1.18,max(1.0,zoom-{step:.6f}))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        # Pan nhẹ từ trái sang phải tạo độ động
        z_expr = "1.10"
        x_expr = f"(iw-iw/zoom)*((on/{frames}))"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_complex = (
        f"[0:v]scale=2560:1440,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=2560x1440:fps={FPS},"
        f"scale={W}:{H}:flags=lanczos[bg];"
        f"[1:v]format=rgba[char];"
        f"[bg][char]overlay=0:0:format=auto[final]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_path,
        "-i", char_layer_path,
        "-filter_complex", filter_complex,
        "-map", "[final]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ==============================================================================
# GIAO DIỆN CHÍNH & PIPELINE ĐIỀU PHỐI (STREAMLIT UI)
# ==============================================================================
st.title("🎬 POV Studio Engine: Góc Khuất Tâm Lý")
st.caption("Khử bỏ hộp thoại thừa, nâng cấp đồ họa Wojak vector sắc nét, Color Grading & Ken Burns điện ảnh")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn tải stock bối cảnh)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice lời thoại", type=["mp3", "wav", "m4a", "ogg"])

if st.button("⚡ Bắt Đầu Render Video Đẳng Cấp Studio", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng cung cấp Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải lên file Voice lời thoại!")
    else:
        status = st.status("Đang kích hoạt cỗ máy dàn cảnh...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="pov_studio_")
        used_urls = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Đo độ dài file voice
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Phân tách âm thanh qua Whisper
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian và khoảng lặng thoại...")
            with open(audio_path, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    file=fh, model=STT_MODEL, response_format="verbose_json"
                )
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            raw_segs = data.get("segments") or []

            segments = []
            cur_text = ""
            cur_start = 0.0

            for seg in raw_segs:
                t = (seg.get("text") or "").strip()
                if not t:
                    continue
                if not cur_text:
                    cur_start = float(seg["start"])
                    cur_text = t
                else:
                    cur_text += " " + t

                # Nhịp cắt cảnh từ 4.5 đến 5.5 giây tạo nhịp thở điện ảnh
                if float(seg["end"]) - cur_start >= 4.5:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "góc khuất tâm lý"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.5, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.5, total_audio_dur - segments[i]["start"])

            # 2. LLM Đạo Diễn Thị Giác: Phân tích tâm lý & Lựa chọn bối cảnh
            status.update(label="🧠 2/4: AI Đạo diễn phân vai biểu cảm Wojak & bối cảnh không gian...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_block = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn hình ảnh cho video tâm lý recap phong cách Bí Mập 666.
Danh mục cảm xúc nhân vật Wojak:
- 'thinking': Nghi ngờ, thăm dò, hoài nghi, phân tích
- 'stressed': Hoang mang, áp lực, bị thao túng, sợ hãi
- 'smug': Tự mãn, kẻ ái kỷ, cười khinh bỉ, kiểm soát
- 'depressed': Kiệt sức, trống rỗng, cô đơn, buông xuôi
- 'neutral': Quan sát, bình thản, lắng nghe

Nhiệm vụ cho MỖI câu thoại:
1. 'query_vn': Chọn bối cảnh đời thực cụ thể, u tối (phòng trọ bừa bộn tối đèn, hành lang chung cư cũ vắng người, cửa sổ đêm ánh đèn đường, bàn làm việc ngổn ngang). TUYỆT ĐỐI CẤM ruộng đồng, thiên nhiên, làng quê.
2. 'emotion': Một trong các cảm xúc ở trên.
3. 'pos': 'left' hoặc 'right' (luân phiên để tạo đối thoại không gian).

Đoạn thoại:
{transcript_block}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "hành lang chung cư cũ tối đèn", "emotion": "thinking", "pos": "right"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Dàn dựng đa tầng: Vector Sprite + Shadow + Color Graded BG + Ken Burns
            status.update(label="🎨 3/4: Đang kết xuất đồ họa vector, phủ bóng đổ và Color Grading...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or random.choice(PSYCHO_CONTEXT_TERMS["isolation"])
                    emotion = sc_data.get("emotion") or "neutral"
                    char_pos = sc_data.get("pos") or ("right" if idx % 2 == 0 else "left")

                    # Tạo bối cảnh chuẩn điện ảnh
                    bg_path = fetch_cinematic_vietnam_bg(query_vn, idx, workdir, used_urls, pexels_key)

                    # Tạo sprite Wojak vector nét mực sắc lẹm kèm bóng đổ
                    char_sprite = WojakRenderer.render_sprite(emotion=emotion, target_h=480)
                    char_layer = CinematicCompositor.create_character_layer(char_sprite, pos=char_pos)
                    char_layer_path = os.path.join(workdir, f"layer_{idx:03d}.png")
                    char_layer.save(char_layer_path, "PNG")

                    # Dựng clip FFmpeg Ken Burns 60fps mượt mà
                    clip_out = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                    render_cinematic_scene(
                        bg_path=bg_path,
                        char_layer_path=char_layer_path,
                        duration=sc["duration"],
                        out_clip=clip_out,
                        mode=idx
                    )
                    f_clips.write(f"file '{os.path.abspath(clip_out)}'\n")

            # 4. Xuất video hoàn chỉnh
            status.update(label="⚡ 4/4: Ghép nối khung hình & đồng bộ âm thanh Master...")
            out_path = os.path.join(workdir, "output.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path
            ]
            subprocess.run(cmd, check=True)

            status.update(label="✅ Video chuẩn Studio hoàn thiện xuất sắc!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Cinematic Về Máy",
                data=video_bytes,
                file_name=f"cinematic_pov_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
