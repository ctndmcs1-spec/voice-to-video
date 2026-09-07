# -*- coding: utf-8 -*-
import os
import re
import json
import random
import shutil
import subprocess
import tempfile
import time
import urllib.parse

import streamlit as st
import requests
from PIL import Image, ImageOps, ImageDraw
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Auto Wojak POV Engine", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"

st.title("🎬 Cỗ Máy Dựng Phim POV Tự Động 100%")
st.caption("Tự cào bối cảnh đời thực, tự tìm meme Wojak theo cảm xúc & tự đục nền trong suốt")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh", type=["mp3", "wav", "m4a", "ogg"])

WOJAK_EMOTION_QUERIES = {
    "neutral": "wojak standing transparent png",
    "thinking": "wojak thinking hand on chin transparent png",
    "shocked": "wojak shocked screaming transparent png",
    "stressed": "wojak stressed holding head transparent png",
    "depressed": "doomer depressed smoking transparent png",
    "smug": "smug wojak smiling transparent png",
    "pointing": "wojak pointing transparent png"
}

def make_white_transparent(img_path: str, out_path: str) -> str:
    """Quét và biến mọi pixel màu trắng/xám sáng thành trong suốt (Alpha = 0)"""
    try:
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGBA")
            datas = img.getdata()
            new_data = []
            for item in datas:
                # Nếu pixel gần trắng (RGB > 220) thì biến thành trong suốt
                if item[0] > 220 and item[1] > 220 and item[2] > 220:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)
            img.save(out_path, "PNG")
            return out_path
    except Exception:
        return img_path

def fetch_wojak_character(emotion: str, idx: int, workdir: str) -> str:
    """Tự động cào ảnh Wojak theo biểu cảm từ mạng và đục nền trong suốt"""
    raw_dest = os.path.join(workdir, f"raw_wojak_{idx:03d}.png")
    clean_dest = os.path.join(workdir, f"clean_wojak_{idx:03d}.png")
    query = WOJAK_EMOTION_QUERIES.get(emotion, WOJAK_EMOTION_QUERIES["neutral"])
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    downloaded = False

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))
            for r in results:
                img_url = r.get("image")
                if img_url and img_url.startswith("http"):
                    try:
                        resp = requests.get(img_url, headers=headers, timeout=5)
                        if resp.status_code == 200 and len(resp.content) > 10000:
                            with open(raw_dest, "wb") as f:
                                f.write(resp.content)
                            downloaded = True
                            break
                    except Exception:
                        continue
    except Exception:
        pass

    if downloaded:
        return make_white_transparent(raw_dest, clean_dest)

    # Fallback tự vẽ Wojak nét vẽ cơ bản nếu rớt mạng
    fallback_img = Image.new("RGBA", (360, 500), (0, 0, 0, 0))
    d = ImageDraw.Draw(fallback_img)
    d.ellipse([90, 40, 270, 220], fill=(245, 235, 225), outline=(15, 15, 15), width=6)
    d.rectangle([110, 220, 250, 480], fill=(40, 45, 55), outline=(15, 15, 15), width=6)
    fallback_img.save(clean_dest, "PNG")
    return clean_dest

def crawl_vietnam_bg(query_vn: str, idx: int, workdir: str, used_urls: set, p_key: str) -> str:
    """Cào ảnh bối cảnh Việt Nam, có cơ chế dự phòng chống màn đen"""
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    search_queries = [
        f"{query_vn} phòng trọ chung cư hà nội",
        f"{query_vn} đời sống thực tế việt nam",
        query_vn
    ]

    for q in search_queries:
        if downloaded:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(q, region="vn-vi", max_results=6))
                for r in results:
                    img_url = r.get("image")
                    if img_url and img_url.startswith("http") and img_url not in used_urls:
                        try:
                            resp = requests.get(img_url, headers=headers, timeout=5)
                            if resp.status_code == 200 and len(resp.content) > 25000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                with Image.open(dest) as t_img:
                                    if t_img.size[0] >= 500:
                                        used_urls.add(img_url)
                                        downloaded = True
                                        break
                        except Exception:
                            continue
        except Exception:
            continue

    if not downloaded:
        fallback_term = "dark moody room apartment window"
        if p_key and p_key.strip():
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(fallback_term)}&per_page=6&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
                if r.ok and r.json().get("photos"):
                    u = r.json()["photos"][0]["src"]["large2x"]
                    with open(dest, "wb") as f:
                        f.write(requests.get(u, timeout=8).content)
                    downloaded = True
            except Exception:
                pass

    if not downloaded:
        img = Image.new('RGB', (W, H), color=(22, 25, 33))
        img.save(dest, "JPEG")

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        pass

    return dest

def render_multi_layer_scene(bg_img: str, char_png: str, dialog_text: str, duration: float, out_clip: str, pos: str = "right", mode: int = 0):
    """FFmpeg ghép 3 tầng: Nền Ken Burns + Wojak trong suốt + Hộp thoại Visual Novel bo viền"""
    frames = max(25, int(duration * FPS))
    step = 0.14 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.14)"
    else:
        z_expr = f"if(eq(on,1),1.14,max(1.0,zoom-{step:.6f}))"

    char_x = "W-w-50" if pos == "right" else "50"
    char_y = "H-h"

    # Làm sạch text tuyệt đối: thay dấu phẩy, hai chấm, nháy đơn để không làm vỡ cú pháp FFmpeg
    safe_text = dialog_text.replace("'", "").replace('"', '').replace(":", " -").replace(",", " -")[:65]

    filter_complex = (
        f"[0:v]scale=2560:1440,"
        f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=2560x1440:fps={FPS},"
        f"scale={W}:{H}:flags=lanczos[bg];"
        f"[1:v]format=rgba,scale=-1:468[char];"
        f"[bg][char]overlay={char_x}:{char_y}[comp];"
        f"[comp]drawbox=x=60:y=H-115:w=W-120:h=85:color=black@0.85:t=fill,"
        f"drawbox=x=60:y=H-115:w=W-120:h=85:color=orange@0.9:t=3,"
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize=28:x=(W-text_w)/2:y=H-80[final]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_img,
        "-i", char_png,
        "-filter_complex", filter_complex,
        "-map", "[final]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

if st.button("⚡ Bắt Đầu Dựng Video Wojak Tự Động", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        status = st.status("Đang kích hoạt cỗ máy dựng video tự động...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="auto_wojak_")
        used_urls = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian câu thoại...")
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

                if float(seg["end"]) - cur_start >= 4.2:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "câu chuyện tâm lý"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.0, total_audio_dur - segments[i]["start"])

            # 2. Phân tích kịch bản: Gán cảm xúc Wojak + Bối cảnh đời thực
            status.update(label="🧠 2/4: AI chọn biểu cảm Wojak & bối cảnh tâm lý đời thực...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:85]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn video recap tâm lý phong cách Bí Mập 666.
Danh sách cảm xúc Wojak:
- 'neutral': lắng nghe, bình thản
- 'thinking': nghi ngờ, phân tích, đặt câu hỏi
- 'shocked': sợ hãi, bàng hoàng, thao túng
- 'stressed': ôm đầu, áp lực, bế tắc
- 'depressed': kiệt sức, buồn bã
- 'smug': tự mãn, mỉa mai, kẻ ái kỷ
- 'pointing': chỉ trích, bóc trần

Trích xuất cho MỖI dòng:
1. 'query_vn': Từ khóa bối cảnh đời thực cụ thể (phòng trọ nhỏ, hành lang chung cư, góc làm việc tối, cửa sổ đêm). CẤM ruộng đồng, phong cảnh thiên nhiên.
2. 'emotion': Một trong các cảm xúc Wojak ở trên.
3. 'pos': 'left' hoặc 'right'.

Đoạn thoại:
{transcript_text}

Trả về JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "góc phòng trọ bừa bộn tối đèn", "emotion": "thinking", "pos": "right"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.25
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Tự động cào ảnh Wojak, đục nền & dựng cảnh 3 lớp
            status.update(label="🎨 3/4: Đang tự cào Wojak, tự đục nền trong suốt & ghép cảnh...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "căn phòng tối tĩnh lặng"
                    emotion = sc_data.get("emotion") or "neutral"
                    char_pos = sc_data.get("pos") or ("right" if idx % 2 == 0 else "left")

                    bg_img = crawl_vietnam_bg(query_vn, idx, workdir, used_urls, pexels_key)
                    char_png = fetch_wojak_character(emotion, idx, workdir)
                    clip_out = os.path.join(workdir, f"clip_{idx:03d}.mp4")

                    render_multi_layer_scene(
                        bg_img=bg_img,
                        char_png=char_png,
                        dialog_text=sc["text"],
                        duration=sc["duration"],
                        out_clip=clip_out,
                        pos=char_pos,
                        mode=idx
                    )
                    f_clips.write(f"file '{os.path.abspath(clip_out)}'\n")

            # 4. Xuất video
            status.update(label="⚡ 4/4: Ghép nối video và đồng bộ audio hoàn chỉnh...")
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

            status.update(label="✅ Video Wojak Visual Novel hoàn thành 100%!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"wojak_story_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
