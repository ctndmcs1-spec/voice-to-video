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
from PIL import Image, ImageOps
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Production Story Video Engine", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🎬 Production Video Engine: Kể Chuyện Đời Thực & Tâm Lý")
st.caption("Khử triệt để màn đen, chuẩn ngữ cảnh tâm lý, Ken Burns 2K và xen kẽ B-roll không trùng")

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Khuyên dùng để lấy B-roll HD)", type="password", placeholder="Nhập key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice lời thoại", type=["mp3", "wav", "m4a", "ogg"])

# Kho dự phòng an toàn (Không bao giờ để video bị màn đen)
SAFE_FALLBACK_TERMS = [
    "lonely person dark room shadow",
    "man looking out window night city",
    "stressed face dark cinematic lighting",
    "empty dark room chair table lamp",
    "silhouette walking alone street night",
    "person sitting alone thinking moody"
]

def fetch_safe_backup_image(query: str, idx: int, p_key: str, workdir: str, used_urls: set) -> str:
    """Tải ảnh chất lượng cao dự phòng khi cào mạng nội địa thất bại, chặn đứng màn đen"""
    dest = os.path.join(workdir, f"backup_{idx:03d}.jpg")
    term = random.choice(SAFE_FALLBACK_TERMS) if not query else query

    if p_key and p_key.strip():
        try:
            url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(term)}&per_page=10&page={(idx % 3) + 1}&orientation=landscape"
            r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=6)
            if r.ok and r.json().get("photos"):
                for p in r.json()["photos"]:
                    u = p["src"]["large2x"]
                    if u not in used_urls:
                        used_urls.add(u)
                        data = requests.get(u, timeout=8).content
                        with open(dest, "wb") as f:
                            f.write(data)
                        return dest
        except Exception:
            pass

    # Nếu không có key Pexels: Lấy từ Lexica theo phong cách điện ảnh tối màu
    try:
        l_url = f"https://lexica.art/api/v1/search?q={urllib.parse.quote(term + ' moody cinematic dark room shadows 35mm')}"
        r = requests.get(l_url, timeout=6)
        if r.ok and r.json().get("images"):
            for img_obj in r.json()["images"]:
                u = img_obj["src"]
                if u not in used_urls:
                    used_urls.add(u)
                    data = requests.get(u, timeout=8).content
                    with open(dest, "wb") as f:
                        f.write(data)
                    return dest
    except Exception:
        pass

    # Trường hợp hy hữu: Tạo gradient nền điện ảnh (tuyệt đối không để đen thui)
    img = Image.new('RGB', (W, H), color=(22, 25, 33))
    img.save(dest, "JPEG")
    return dest

def crawl_vietnam_media(query_vn: str, idx: int, workdir: str, used_urls: set, p_key: str) -> str:
    """Cào ảnh thực tế từ mạng Việt Nam, kiểm tra kích thước và loại bỏ ảnh lỗi"""
    dest = os.path.join(workdir, f"media_{idx:03d}.jpg")
    downloaded = False
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    search_list = [
        f"{query_vn} đời sống thực tế",
        f"{query_vn} chụp thực tế",
        query_vn
    ]

    for q in search_list:
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
                            # Bỏ qua các file ảnh quá bé hoặc thumbnail rác (< 25KB)
                            if resp.status_code == 200 and len(resp.content) > 25000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                
                                # Kiểm tra mở thử bằng PIL
                                with Image.open(dest) as test_img:
                                    w_raw, h_raw = test_img.size
                                    if w_raw >= 600 and h_raw >= 400:
                                        used_urls.add(img_url)
                                        downloaded = True
                                        break
                        except Exception:
                            continue
        except Exception:
            continue

    # Nếu cào thất bại: Kích hoạt tầng dự phòng chất lượng cao ngay lập tức
    if not downloaded:
        dest = fetch_safe_backup_image(query_vn, idx, p_key, workdir, used_urls)

    try:
        with Image.open(dest) as img:
            fitted = ImageOps.fit(img.convert("RGB"), (W, H), Image.LANCZOS)
            fitted.save(dest, "JPEG", quality=92)
    except Exception:
        dest = fetch_safe_backup_image(query_vn, idx, p_key, workdir, used_urls)

    return dest

def fetch_broll_clip(query_en: str, idx: int, duration: float, p_key: str, workdir: str, used_vid_ids: set) -> str:
    """Lấy video clip chuyển động 5s, cắt ghép chuẩn và chống lặp ID 100%"""
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False

    search_terms = [query_en, "night city walking silhouette", "clock ticking timelapse", "traffic moving blur"]

    for term in search_terms:
        if downloaded:
            break
        try:
            url = f"{PEXELS_VIDEO_URL}?query={urllib.parse.quote(term)}&per_page=8&orientation=landscape"
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
        filter_str = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p"
        cmd = [
            "ffmpeg", "-y", "-ss", "0", "-i", raw_vid,
            "-t", f"{duration:.3f}",
            "-vf", filter_str,
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast",
            clip_dest
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(raw_vid):
            os.remove(raw_vid)
        return clip_dest

    return None

def create_kenburns_clip(img_path: str, duration: float, out_clip: str, mode: int = 0):
    """Zoom/Lia máy 2K nội bộ, khử 100% hiện tượng rung giật pixel"""
    frames = max(25, int(duration * FPS))
    step = 0.16 / frames

    if mode % 2 == 0:
        z_expr = f"min(zoom+{step:.6f},1.16)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"if(eq(on,1),1.16,max(1.0,zoom-{step:.6f}))"
        x_expr = "iw/2-(iw/zoom/2)"
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
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

if st.button("⚡ Bắt Đầu Dựng Video Hoàn Chỉnh", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file âm thanh lên trước!")
    else:
        status = st.status("Đang khởi động cỗ máy dựng video cao cấp...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="pro_engine_")
        used_urls = set()
        used_vid_ids = set()

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)

            client = Groq(api_key=groq_key.strip())

            # 1. Bóc tách âm thanh & Đồng bộ nhịp thoại
            status.update(label="🎙️ 1/4: Whisper phân tích mốc thời gian & khoảng lặng câu thoại...")
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

                # Đảm bảo mỗi cảnh duy trì 4.2 - 5.5 giây để đủ nhịp tiếp thu thị giác
                if float(seg["end"]) - cur_start >= 4.2:
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

            # 2. Ép LLM chuyển đổi khái niệm trừu tượng thành hình ảnh đời thực/tâm lý cụ thể
            status.update(label="🧠 2/4: AI đạo diễn lên bối cảnh tâm lý điện ảnh (Chặn đứng lạc đề)...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])
                prompt = f"""Bạn là đạo diễn hình ảnh cho video tâm lý và đời sống xã hội thực tế.
Nhiệm vụ: Chuyển các câu thoại sau thành từ khóa hình ảnh đời thực hoặc ẩn dụ điện ảnh cụ thể (3-5 từ).
QUY TẮC CỐT LÕI:
- CẤM TUYỆT ĐỐI phong cảnh thiên nhiên, ruộng bậc thang, làng quê nón lá, đồng lúa.
- Với các đoạn nói về thao túng, ái kỷ, kiệt sức, tội đồ: chuyển thành vật thể/bối cảnh cụ thể: 'người cô đơn trong phòng tối', 'bóng người qua khe cửa', 'người đàn ông nhìn qua cửa sổ đêm', 'bàn làm việc ngổn ngang ban đêm', 'người ngồi ôm đầu tuyệt vọng'.
- Với các đoạn đời sống: 'phòng trọ nhỏ gác xép', 'xe máy dừng ngã tư đèn đỏ', 'hành lang chung cư cũ vắng người'.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT JSON:
{{"scenes": [{{"index": {b_start}, "query_vn": "người cô đơn trong phòng tối", "query_en": "lonely person dark room shadow"}}]}}"""

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

            # 3. Gom ảnh/video xen kẽ và render
            status.update(label="🎬 3/4: Đang gom tư liệu thực tế & dựng chuyển động Ken Burns...")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "người ngồi trầm ngâm trong phòng tối"
                    query_en = sc_data.get("query_en") or "silhouette person looking out window night"
                    dur = sc["duration"]

                    clip_path = None
                    # Đan xen B-roll: Cảnh thứ 3 hoặc thứ 4 lấy video động (nếu có key Pexels)
                    if idx % 3 == 1 and pexels_key:
                        clip_path = fetch_broll_clip(query_en, idx, dur, pexels_key, workdir, used_vid_ids)

                    # Cảnh ảnh tĩnh: cào mạng Việt Nam hoặc fallback sang ảnh stock moody chất lượng cao
                    if not clip_path:
                        img_path = crawl_vietnam_media(query_vn, idx, workdir, used_urls, pexels_key)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, dur, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video hoàn thiện
            status.update(label="⚡ 4/4: Ghép video và đồng bộ audio hoàn chỉnh...")
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

            status.update(label="✅ Video hoàn thiện thành công với chuẩn chất lượng mới!", state="complete")

            with open(out_path, "rb") as vid_file:
                video_bytes = vid_file.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Tải Video Về Máy",
                data=video_bytes,
                file_name=f"cinematic_story_{int(time.time())}.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            status.update(label=f"❌ Thất bại: {str(e)}", state="error")
            st.error(f"Chi tiết lỗi: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
