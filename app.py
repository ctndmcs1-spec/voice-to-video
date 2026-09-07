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

st.set_page_config(page_title="Visual Novel POV Engine Pro", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🎬 Cỗ Máy Dựng Phim POV Hoạt Họa 2D")
st.caption("Tự động tách nền trắng, dàn cảnh 3 lớp: Bối cảnh Việt Nam + Nhân vật hoạt họa + Hộp thoại")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn B-roll HD)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh", type=["mp3", "wav", "m4a", "ogg"])

POSES = [
    "nhan_vat_binh_thuong.png",
    "nhan_vat_suy_nghi.png",
    "nhan_vat_soc.png",
    "nhan_vat_kiet_suc.png",
    "nhan_vat_chi_tay.png"
]

def make_white_transparent(img_path: str, temp_dir: str) -> str:
    """Tự động chuyển pixel trắng hoặc gần trắng thành trong suốt (RGBA alpha=0)"""
    try:
        base_name = os.path.basename(img_path)
        out_path = os.path.join(temp_dir, f"trans_{base_name}")
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGBA")
            datas = img.getdata()
            new_data = []
            for item in datas:
                # Ngưỡng RGB > 225 coi là nền trắng sáng
                if item[0] > 225 and item[1] > 225 and item[2] > 225:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)
            img.save(out_path, "PNG")
            return out_path
    except Exception:
        return img_path

def init_mock_characters(assets_dir: str):
    """Tạo phôi nhân vật hoạt họa viền đậm dự phòng nếu thư mục chưa có file ảnh"""
    os.makedirs(assets_dir, exist_ok=True)
    palette = {
        "nhan_vat_binh_thuong.png": ((245, 230, 215), "NEUTRAL"),
        "nhan_vat_suy_nghi.png": ((220, 235, 255), "THINKING"),
        "nhan_vat_soc.png": ((255, 215, 215), "SHOCKED"),
        "nhan_vat_kiet_suc.png": ((215, 215, 225), "EXHAUSTED"),
        "nhan_vat_chi_tay.png": ((255, 245, 205), "POINTING")
    }
    for fname, (bg_col, tag) in palette.items():
        p = os.path.join(assets_dir, fname)
        if not os.path.exists(p):
            img = Image.new("RGBA", (380, 560), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.ellipse([90, 40, 290, 240], fill=bg_col, outline=(15, 15, 15), width=7)
            d.rectangle([120, 110, 175, 150], outline=(15, 15, 15), width=6)
            d.rectangle([205, 110, 260, 150], outline=(15, 15, 15), width=6)
            d.line([175, 130, 205, 130], fill=(15, 15, 15), width=6)
            d.rectangle([110, 240, 270, 520], fill=(35, 40, 50), outline=(15, 15, 15), width=7)
            d.text((135, 340), tag, fill=(255, 255, 255))
            img.save(p, "PNG")

def crawl_vietnam_bg(query_vn: str, idx: int, workdir: str, used_urls: set, p_key: str) -> str:
    """Cào ảnh bối cảnh Việt Nam, có cơ chế dự phòng chống màn đen"""
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    search_queries = [
        f"{query_vn} phòng trọ chung cư",
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
        fallback_term = "dark room window moody apartment interior"
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
        img = Image.new('RGB', (W, H), color=(25, 28, 36))
        img.save(dest, "JPEG")

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=90)
    except Exception:
        pass

    return dest

def render_multi_layer_scene(bg_img: str, char_png: str, dialog_text: str, duration: float, out_clip: str, pos: str = "right", mode: int = 0):
    """FFmpeg ghép 3 tầng: Nền Ken Burns 2K + Nhân vật PNG trong suốt + Hộp thoại Visual Novel"""
    frames = max(25, int(duration * FPS))
    step = 0.14 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.14)"
    else:
        z_expr = f"if(eq(on,1),1.14,max(1.0,zoom-{step:.6f}))"

    char_x = "W-w-50" if pos == "right" else "50"
    char_y = "H-h"
    clean_text = dialog_text.replace("'", "").replace('"', '').replace(":", " -")[:65]

    filter_complex = (
        f"[0:v]scale=2560:1440,zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=2560x1440:fps={FPS},scale={W}:{H}:flags=lanczos[bg];"
        f"[1:v]scale=-1:468[char];"
        f"[bg][char]overlay={char_x}:{char_y}[comp];"
        f"[comp]drawbox=x=60:y=H-115:w=W-120:h=85:color=0x111319@0.85:t=fill,"
        f"drawbox=x=60:y=H-115:w=W-120:h=85:color=0xf39c12@0.9:t=3,"
        f"drawtext=text='{clean_text}':fontcolor=white:fontsize=28:x=(W-text_w)/2:y=H-82[final]"
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

if st.button("⚡ Bắt Đầu Dựng Video Visual Novel", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        status = st.status("Đang kích hoạt cỗ máy dàn cảnh 3 lớp...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="vn_final_")
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        init_mock_characters(assets_dir)
        used_urls = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh qua Whisper
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian từng câu thoại...")
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

                if float(seg["end"]) - cur_start >= 4.0:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                end_time = float(raw_segs[-1]["end"]) if raw_segs else total_audio_dur
                segments.append({"start": cur_start, "end": end_time, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "câu chuyện tâm lý đời thực"})

            for i in range(len(segments)):
                if i < len(segments) - 1:
                    segments[i]["duration"] = max(2.0, segments[i+1]["start"] - segments[i]["start"])
                else:
                    segments[i]["duration"] = max(2.0, total_audio_dur - segments[i]["start"])

            # 2. Phân tích kịch bản bằng LLM
            status.update(label="🧠 2/4: AI đạo diễn phân vai biểu cảm nhân vật & bối cảnh...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:85]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn phim hoạt họa tâm lý POV đời sống Việt Nam.
Danh sách nhân vật:
- 'nhan_vat_binh_thuong.png' (lắng nghe, đứng nhìn)
- 'nhan_vat_suy_nghi.png' (nghi ngờ, phân tích)
- 'nhan_vat_soc.png' (bị thao túng, sợ hãi)
- 'nhan_vat_kiet_suc.png' (áp lực, mệt mỏi)
- 'nhan_vat_chi_tay.png' (đối chất, bóc trần)

Dựa trên câu thoại, trích xuất cho MỖI đoạn:
1. Từ khóa bối cảnh đời thực Việt Nam (phòng trọ gác xép, hành lang chung cư cũ, ngã tư đèn đỏ, góc bàn làm việc tối, cửa sổ đêm). TUYỆT ĐỐI CẤM đồng lúa, ruộng bậc thang.
2. Tên file nhân vật tương ứng.
3. Vị trí: 'left' hoặc 'right'.

Đoạn thoại:
{transcript_text}

Trả về JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "góc phòng trọ bừa bộn tối đèn", "pose": "nhan_vat_suy_nghi.png", "pos": "right"}}
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

            # 3. Dựng clip 3 lớp đa tầng
            status.update(label="🎨 3/4: Đang tách nền nhân vật, ghép nền Ken Burns và hộp thoại...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "căn phòng tối tĩnh lặng"
                    pose_name = sc_data.get("pose") or POSES[idx % len(POSES)]
                    char_pos = sc_data.get("pos") or ("right" if idx % 2 == 0 else "left")

                    raw_char_file = os.path.join(assets_dir, pose_name)
                    if not os.path.exists(raw_char_file):
                        raw_char_file = os.path.join(assets_dir, POSES[0])

                    # Tự động tách nền trắng thành trong suốt
                    clean_char_file = make_white_transparent(raw_char_file, workdir)

                    bg_img = crawl_vietnam_bg(query_vn, idx, workdir, used_urls, pexels_key)
                    clip_out = os.path.join(workdir, f"clip_{idx:03d}.mp4")

                    render_multi_layer_scene(
                        bg_img=bg_img,
                        char_png=clean_char_file,
                        dialog_text=sc["text"],
                        duration=sc["duration"],
                        out_clip=clip_out,
                        pos=char_pos,
                        mode=idx
                    )
                    f_clips.write(f"file '{os.path.abspath(clip_out)}'\n")

            # 4. Xuất video hoàn thiện
            status.update(label="⚡ 4/4: Ghép nối video và đồng bộ audio...")
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

            status.update(label="✅ Video Visual Novel hoàn thiện xuất sắc!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"pov_vn_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
