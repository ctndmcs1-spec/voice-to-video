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
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Studio POV Production Master", page_icon="🎬", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🎬 Studio POV Video Engine Pro")
st.caption("Đồng bộ khung hình tuyệt đối, loại bỏ hoàn toàn màn hình đen, tối ưu nội dung ngữ cảnh")

genre_mode = st.selectbox(
    "Chọn phong cách & Tone màu chủ đạo của Video:",
    [
        "Tài chính / Khởi nghiệp & Kịch tính (Corporate / Hustle)",
        "Nghề nghiệp / Tươi sáng & Động lực (Bright Career)",
        "Tâm lý / Góc khuất & U tối (Dark Moody POV)",
        "Đời sống thường nhật & Hoài niệm (Vintage Lofi Life)"
    ]
)

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Khuyên dùng để lấy video B-roll HD)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh", type=["mp3", "wav", "m4a", "ogg"])

# Kho dự phòng an toàn tuyệt đối theo phong cách Corporate / Finance
STUDIO_SAFE_ASSETS = [
    "modern office business district night",
    "financial audit report numbers documents",
    "stressed employee working desk late night",
    "corporate meeting room presentation",
    "laptop keyboard typing financial graph",
    "empty modern office hallway perspective"
]

def apply_genre_color_grading(img: Image.Image, genre: str) -> Image.Image:
    """Xử lý màu sắc phù hợp với tone phân đoạn"""
    if "Corporate" in genre:
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.18)
        r, g, b = graded.split()
        b = b.point(lambda i: min(255, int(i * 1.06)))
        return Image.merge("RGB", (r, g, b))
    elif "Dark Moody" in genre:
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.22)
        enhancer = ImageEnhance.Brightness(graded)
        graded = enhancer.enhance(0.88)
        return graded
    elif "Bright Career" in genre:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(1.15)
    else:
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(0.92)
        r, g, b = graded.split()
        r = r.point(lambda i: min(255, int(i * 1.05)))
        return Image.merge("RGB", (r, g, b))

def generate_fallback_canvas(dest_path: str, idx: int):
    """Tạo phôi đồ họa studio tối màu nếu toàn bộ nguồn mạng bị chặn, không để màn đen trống"""
    canvas = Image.new("RGB", (W, H), (25, 30, 40))
    d = ImageDraw.Draw(canvas)
    # Vẽ các mảng gradient tạo chiều sâu không gian làm việc
    for y in range(H):
        val = int(25 + (y / H) * 20)
        d.line([(0, y), (W, y)], fill=(val, val + 5, val + 15))
    canvas.save(dest_path, "JPEG", quality=90)

def fetch_matching_image(query_vn: str, query_en: str, idx: int, workdir: str, used_urls: set, p_key: str, genre: str) -> str:
    """Bộ cào ảnh thông minh đa lớp, có chống link rác và fallback nội bộ"""
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Bổ sung tiền tố ngữ cảnh công sở chống lạc đề
    safe_en = f"corporate {query_en}" if "office" not in query_en.lower() and "work" not in query_en.lower() else query_en

    # 1. Tìm qua Pexels nếu có key
    if p_key and p_key.strip():
        for q in [safe_en, query_en]:
            if downloaded:
                break
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(q)}&per_page=6&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
                if r.ok and r.json().get("photos"):
                    for p in r.json()["photos"]:
                        u = p["src"]["large2x"]
                        if u not in used_urls:
                            used_urls.add(u)
                            resp = requests.get(u, timeout=8)
                            if resp.status_code == 200 and len(resp.content) > 30000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                downloaded = True
                                break
            except Exception:
                continue

    # 2. Tìm qua DuckDuckGo
    if not downloaded:
        search_candidates = [f"{safe_en} high quality wallpaper", f"{query_vn} công sở thực tế", safe_en]
        for q in search_candidates:
            if downloaded:
                break
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(q, max_results=6))
                    for r in results:
                        u = r.get("image")
                        if u and u.startswith("http") and u not in used_urls:
                            try:
                                resp = requests.get(u, headers=headers, timeout=5)
                                # Loại bỏ các file rác hoặc banner quá nhỏ
                                if resp.status_code == 200 and len(resp.content) > 35000:
                                    with open(dest, "wb") as f:
                                        f.write(resp.content)
                                    with Image.open(dest) as t_img:
                                        if t_img.size[0] >= 600 and t_img.size[1] >= 400:
                                            used_urls.add(u)
                                            downloaded = True
                                            break
                            except Exception:
                                continue
            except Exception:
                continue

    # 3. Fallback danh mục an toàn
    if not downloaded:
        fb_term = random.choice(STUDIO_SAFE_ASSETS)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(fb_term, max_results=5))
                for r in results:
                    u = r.get("image")
                    if u and u.startswith("http") and u not in used_urls:
                        resp = requests.get(u, headers=headers, timeout=5)
                        if resp.status_code == 200 and len(resp.content) > 25000:
                            with open(dest, "wb") as f:
                                f.write(resp.content)
                            downloaded = True
                            break
        except Exception:
            pass

    if not downloaded:
        generate_fallback_canvas(dest, idx)

    # Chuẩn hóa kích thước và phủ màu
    try:
        with Image.open(dest) as raw_img:
            fitted = ImageOps.fit(raw_img.convert("RGB"), (W, H), Image.LANCZOS)
            graded = apply_genre_color_grading(fitted, genre)
            graded.save(dest, "JPEG", quality=92)
    except Exception:
        generate_fallback_canvas(dest, idx)

    return dest

def fetch_broll_clip(query_en: str, idx: int, target_frames: int, p_key: str, workdir: str, used_vid_ids: set) -> str:
    """Tải clip B-roll và chuẩn hóa chính xác số khung hình tuyệt đối"""
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False

    search_terms = [
        f"corporate {query_en}",
        query_en,
        "office worker typing desk night",
        "financial business meeting modern office"
    ]

    for term in search_terms:
        if downloaded:
            break
        try:
            url = f"{PEXELS_VIDEO_URL}?query={urllib.parse.quote(term)}&per_page=6&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
            if r.ok and r.json().get("videos"):
                for v in r.json()["videos"]:
                    v_id = v.get("id")
                    if v_id and v_id not in used_vid_ids:
                        used_vid_ids.add(v_id)
                        vid_files = v.get("video_files", [])
                        hd_files = [f for f in vid_files if f.get("height", 0) >= 720 and f.get("file_type") == "video/mp4"]
                        target_url = hd_files[0]["link"] if hd_files else vid_files[0]["link"]
                        with requests.get(target_url, stream=True, timeout=15) as stream:
                            with open(raw_vid, "wb") as f_out:
                                shutil.copyfileobj(stream.raw, f_out)
                        downloaded = True
                        break
        except Exception:
            continue

    if downloaded and os.path.exists(raw_vid):
        try:
            filter_str = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p"
            cmd = [
                "ffmpeg", "-y", "-ss", "0", "-i", raw_vid,
                "-vframes", str(target_frames),
                "-vf", filter_str,
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast",
                clip_dest
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(raw_vid):
                os.remove(raw_vid)
            return clip_dest
        except Exception:
            pass

    return None

def create_kenburns_clip(img_path: str, target_frames: int, out_clip: str, mode: int = 0):
    """Render Ken Burns đa trục dựa trên số frame chính xác 100%"""
    frames = max(25, target_frames)
    step = 0.15 / frames
    m = mode % 4

    if m == 0:
        z_expr = f"min(zoom+{step:.6f},1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif m == 1:
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*(on/{frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    elif m == 2:
        z_expr = f"if(eq(on,1),1.15,max(1.0,zoom-{step:.6f}))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = "1.15"
        x_expr = f"(iw-iw/zoom)*(1-on/{frames})"
        y_expr = "ih/2-(ih/zoom/2)"

    filter_complex = (
        f"scale=2560:1440,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=2560x1440:fps={FPS},"
        f"scale={W}:{H}:flags=lanczos,"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", filter_complex,
        "-vframes", str(frames),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ==============================================================================
# QUY TRÌNH SẢN XUẤT CHÍNH
# ==============================================================================
if st.button("⚡ Bắt Đầu Dựng Video Thành Phẩm Hoàn Chỉnh", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        status = st.status("Đang chuẩn bị dây chuyền sản xuất video...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="master_prod_")
        used_urls = set()
        used_vid_ids = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Đo đạc thời lượng file gốc chính xác tuyệt đối
            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)
            total_required_frames = int(round(total_audio_dur * FPS))

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh (Nén mono 16kHz chống 413 Payload Too Large)
            status.update(label="🎙️ 1/4: Đang tối ưu dung lượng & Whisper phân tích mốc thời gian...")
            compressed_audio = os.path.join(workdir, "whisper_input.mp3")
            compress_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-vn", "-ar", "16000", "-ac", "1", "-b:a", "48k",
                compressed_audio
            ]
            subprocess.run(compress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with open(compressed_audio, "rb") as fh:
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

                # Nhịp ngắt cảnh tiêu chuẩn 4.5s
                if float(seg["end"]) - cur_start >= 4.5:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                segments.append({"start": cur_start, "end": total_audio_dur, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "góc khuất nghề nghiệp"})

            # Tính toán phân bổ Frame tuyệt đối
            accumulated_frames = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i+1]["start"] - segments[i]["start"]
                    f_count = int(round(seg_dur * FPS))
                    segments[i]["target_frames"] = max(25, f_count)
                    accumulated_frames += segments[i]["target_frames"]
                else:
                    # Cảnh cuối gánh trọn vẹn số frame còn lại, bù đắp mọi sai số
                    remaining = total_required_frames - accumulated_frames
                    segments[i]["target_frames"] = max(25, remaining)

            # 2. AI Đạo diễn bóc tách từ khóa với bộ lọc ngữ cảnh chuyên sâu
            status.update(label="🧠 2/4: AI đạo diễn trích xuất bối cảnh chuẩn ngữ cảnh công sở...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn hình ảnh cho video phân tích góc khuất nghề nghiệp: "{genre_mode}".
Nội dung video nói về ngành KIỂM TOÁN, TÀI CHÍNH, DOANH NGHIỆP, ÁP LỰC CÔNG SỞ.

QUY TẮC CỐT TỬ ĐỂ KHÔNG BỊ LẠC ĐỀ:
1. Luôn neo trong môi trường văn phòng, tài chính: bàn làm việc, tài liệu số liệu, họp cổ đông, sếp và nhân viên, ký kết hợp đồng, máy tính kế toán, hành lang công ty, cà phê đêm làm việc.
2. TUYỆT ĐỐI CẤM hiểu theo nghĩa đen ngây thơ:
   - Nói "ngàn cân" -> KHÔNG ĐƯỢC lấy tập gym/deadlift. Phải lấy: 'người ngồi ôm đầu trước đống tài liệu dày'.
   - Nói "nộp đơn" -> KHÔNG ĐƯỢC lấy đơn ly hôn. Phải lấy: 'đơn xin việc, CV văn phòng'.
   - Nói "vừa đấm vừa xoa" -> KHÔNG ĐƯỢC lấy hoạt động ngoài trời/bèo tây. Phải lấy: 'cuộc họp thương thuyết đối tác'.
   - Nói "dòng máu tài chính/gian lận" -> KHÔNG ĐƯỢC lấy dạy nhạc/trường học. Phải lấy: 'báo cáo tài chính, biểu đồ tiền tệ'.
3. CẢNH ĐẦU TIÊN [0]: Bắt buộc lấy tòa nhà chọc trời ban đêm rực rỡ ánh đèn ('city skyscraper night lights').
4. CẢNH CUỐI CÙNG: Bắt buộc lấy lời cảm ơn, ghi chép sổ tay hoặc màn hình làm việc kết thúc ngày.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "tòa nhà chọc trời ban đêm rực rỡ", "query_en": "city skyscraper night lights"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.15
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Dựng cảnh 40% Video B-roll + 60% Ảnh tĩnh (Không frame lỗi)
            status.update(label="🎬 3/4: Đang kết xuất clip theo mốc frame chính xác...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "hồ sơ tài chính văn phòng"
                    query_en = sc_data.get("query_en") or "corporate financial office work"
                    t_frames = sc["target_frames"]

                    clip_path = None
                    is_video_slot = (idx % 5 in [1, 3]) and (idx != len(segments) - 1)
                    if is_video_slot and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, t_frames, pexels_key, workdir, used_vid_ids)

                    if not clip_path:
                        img_path = fetch_matching_image(query_vn, query_en, idx, workdir, used_urls, pexels_key, genre_mode)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video hoàn thiện khớp khung hình
            status.update(label="⚡ 4/4: Ghép video và nén xuất Master...", state="running")
            out_path = os.path.join(workdir, "output.mp4")

            # Sử dụng cờ muxing chuẩn xác tuyệt đối không chém đuôi audio
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", clips_txt,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-t", f"{total_audio_dur:.3f}",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                out_path
            ]
            subprocess.run(cmd, check=True)

            status.update(label="✅ Thành phẩm xuất bản đã hoàn thành hoàn hảo!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Hoàn Chỉnh Lên Kênh YouTube",
                data=video_bytes,
                file_name=f"youtube_master_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
