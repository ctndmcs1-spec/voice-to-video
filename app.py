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
from PIL import Image, ImageOps, ImageEnhance
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Studio POV Master Engine - Adaptive", page_icon="🎬", layout="centered")

W, H = 1280, 720
FPS = 25
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"

st.title("🎬 Studio POV Master Engine Pro")
st.caption("Chế độ Tự do bám sát 100% ngữ cảnh Voice - Chống lệch đề & Chống lối mòn")

render_engine = st.radio(
    "Chọn phương thức tạo hình ảnh:",
    [
        "📷 Stock Footage: Người thật / B-roll (Pexels + DuckDuckGo + Wikimedia)",
        "🎨 AI Gen: Webtoon / Manhwa POV (Tự vẽ 100% bằng AI Flux, chuẩn tỷ lệ nhân vật)"
    ]
)

genre_mode = st.selectbox(
    "Chọn phong cách / Ngách kịch bản:",
    [
        "Tự do / Thích ứng 100% theo Voice (Adaptive Voice Auto-Match)",
        "POV Bạn bè & Tình huống Xã hội (Social Life / Dating POV)",
        "Đời sống thường nhật & Bụi bặm (Street Life / Realistic)",
        "Học đường / Webtoon Manhwa (Handsome Boy / School POV)",
        "Tâm lý / Góc khuất & Thực tế (Real Life POV)",
        "Tài chính / Khởi nghiệp & Kịch tính (Corporate / Hustle)"
    ]
)

groq_key = st.text_input("Groq API Key (Bắt buộc)", type="password", placeholder="gsk_...")
pexels_key = st.text_input("Pexels API Key (Tùy chọn cho chế độ Stock footage)", type="password", placeholder="Key Pexels...")
audio_file = st.file_uploader("Tải lên file Voice âm thanh (Hỗ trợ MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])

def apply_genre_color_grading(img: Image.Image, genre: str) -> Image.Image:
    if "Webtoon" in genre or "Học đường" in genre:
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(1.08)
        enhancer = ImageEnhance.Contrast(graded)
        return enhancer.enhance(1.05)
    elif "Tâm lý" in genre:
        enhancer = ImageEnhance.Contrast(img)
        graded = enhancer.enhance(1.15)
        enhancer = ImageEnhance.Brightness(graded)
        return enhancer.enhance(0.92)
    elif "Tự do" in genre or "Social Life" in genre:
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(1.06)
        enhancer = ImageEnhance.Contrast(graded)
        return enhancer.enhance(1.06)
    elif "Đời sống" in genre:
        enhancer = ImageEnhance.Color(img)
        graded = enhancer.enhance(1.05)
        enhancer = ImageEnhance.Contrast(graded)
        return enhancer.enhance(1.08)
    else:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.12)

def generate_manhwa_image(query_en: str, idx: int, workdir: str, char_seed: int, genre: str) -> str:
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    style_core = "masterpiece, best quality, clean 2D korean manhwa style, sharp detailed lines, anime aesthetic illustration"
    character_dna = "16-year-old handsome boy, stylish black hair, expressive eyes, school uniform shirt"
    clean_action = re.sub(r'[^a-zA-Z0-9\s,]', '', query_en)

    full_prompt = f"{style_core}, {character_dna}, {clean_action}, cinematic lighting, 4k"
    encoded = urllib.parse.quote(full_prompt)

    gen_url = f"https://image.pollinations.ai/prompt/{encoded}?width={W}&height={H}&model=flux-anime&seed={char_seed}&nologo=true"

    downloaded = False
    for _ in range(2):
        try:
            resp = requests.get(gen_url, headers=headers, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 30000:
                with open(dest, "wb") as f:
                    f.write(resp.content)
                downloaded = True
                break
        except Exception:
            time.sleep(1)

    if not downloaded:
        fallback_url = f"https://picsum.photos/seed/{char_seed + idx}/1280/720"
        try:
            resp = requests.get(fallback_url, timeout=7)
            if resp.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(resp.content)
                downloaded = True
        except Exception:
            pass

    try:
        with Image.open(dest) as raw_img:
            fitted = ImageOps.fit(raw_img.convert("RGB"), (W, H), Image.LANCZOS)
            graded = apply_genre_color_grading(fitted, genre)
            graded.save(dest, "JPEG", quality=92)
    except Exception:
        safe_fallback = Image.new("RGB", (W, H), (25, 30, 45))
        safe_fallback.save(dest, "JPEG", quality=90)

    return dest

def fetch_matching_image(query_vn: str, query_en: str, idx: int, workdir: str, used_urls: set, p_key: str, genre: str, is_english: bool) -> str:
    dest = os.path.join(workdir, f"bg_{idx:03d}.jpg")
    downloaded = False
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    search_en = query_en.strip() if query_en else "everyday scene"

    # Tầng 1: Pexels API
    if p_key and p_key.strip():
        for q in [search_en, f"{search_en} realistic"]:
            if downloaded:
                break
            try:
                url = f"{PEXELS_PHOTO_URL}?query={urllib.parse.quote(q)}&per_page=15&orientation=landscape"
                r = requests.get(url, headers={"Authorization": p_key.strip()}, timeout=5)
                if r.ok and r.json().get("photos"):
                    for p in r.json()["photos"]:
                        u = p["src"]["large2x"]
                        if u not in used_urls:
                            resp = requests.get(u, timeout=7)
                            if resp.status_code == 200 and len(resp.content) > 30000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                used_urls.add(u)
                                downloaded = True
                                break
            except Exception:
                continue

    # Tầng 2: Wikimedia Commons API
    if not downloaded:
        try:
            wiki_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(search_en)}&gsrlimit=10&prop=imageinfo&iiprop=url|size&format=json"
            r = requests.get(wiki_url, headers=headers, timeout=5)
            if r.ok:
                pages = r.json().get("query", {}).get("pages", {})
                for page_id, info in pages.items():
                    img_info = info.get("imageinfo", [{}])[0]
                    u = img_info.get("url")
                    if u and u.lower().endswith(('.jpg', '.jpeg', '.png')) and u not in used_urls:
                        if img_info.get("width", 0) >= 800:
                            resp = requests.get(u, headers=headers, timeout=7)
                            if resp.status_code == 200 and len(resp.content) > 30000:
                                with open(dest, "wb") as f:
                                    f.write(resp.content)
                                used_urls.add(u)
                                downloaded = True
                                break
        except Exception:
            pass

    # Tầng 3: DuckDuckGo Search
    if not downloaded:
        search_list = [f"{search_en} photo", search_en]
        if not is_english and query_vn:
            search_list.append(query_vn)

        ddg_region = "wt-wt" if is_english else "vn-vi"
        for q in search_list:
            if downloaded:
                break
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.images(q, region=ddg_region, max_results=6))
                    for r in results:
                        u = r.get("image")
                        if u and u.startswith("http") and u not in used_urls:
                            try:
                                resp = requests.get(u, headers=headers, timeout=5)
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

    # Tầng 4: Fallback theo đúng từ khóa chính của câu thoại
    if not downloaded:
        words = [w for w in re.findall(r'[a-zA-Z]+', search_en) if len(w) > 3]
        fallback_tag = words[0] if words else "lifestyle"
        fallback_url = f"https://loremflickr.com/1280/720/{urllib.parse.quote(fallback_tag)}?lock={idx + 100}"
        try:
            resp = requests.get(fallback_url, timeout=7)
            if resp.status_code == 200 and len(resp.content) > 20000:
                with open(dest, "wb") as f:
                    f.write(resp.content)
                downloaded = True
        except Exception:
            pass

    try:
        with Image.open(dest) as raw_img:
            fitted = ImageOps.fit(raw_img.convert("RGB"), (W, H), Image.LANCZOS)
            graded = apply_genre_color_grading(fitted, genre)
            graded.save(dest, "JPEG", quality=92)
    except Exception:
        safe_fallback = Image.new("RGB", (W, H), (30, 35, 45))
        safe_fallback.save(dest, "JPEG", quality=90)

    return dest

def fetch_broll_clip(query_en: str, idx: int, target_frames: int, p_key: str, workdir: str, used_vid_ids: set) -> str:
    if not p_key or not p_key.strip():
        return None

    clip_dest = os.path.join(workdir, f"clip_{idx:03d}.mp4")
    raw_vid = os.path.join(workdir, f"raw_{idx:03d}.mp4")
    downloaded = False
    dur = target_frames / FPS

    search_terms = [query_en, f"{query_en} realistic"]

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
                        vid_files = v.get("video_files", [])
                        hd_files = [f for f in vid_files if f.get("height", 0) >= 720 and f.get("file_type") == "video/mp4"]
                        target_url = hd_files[0]["link"] if hd_files else vid_files[0]["link"]
                        with requests.get(target_url, stream=True, timeout=15) as stream:
                            with open(raw_vid, "wb") as f_out:
                                shutil.copyfileobj(stream.raw, f_out)
                        if os.path.exists(raw_vid) and os.path.getsize(raw_vid) > 50000:
                            used_vid_ids.add(v_id)
                            downloaded = True
                            break
        except Exception:
            continue

    if downloaded and os.path.exists(raw_vid):
        try:
            filter_str = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p"
            cmd = [
                "ffmpeg", "-y", "-ss", "0", "-i", raw_vid,
                "-t", f"{dur:.3f}",
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
            if os.path.exists(raw_vid):
                os.remove(raw_vid)

    return None

def create_kenburns_clip(img_path: str, target_frames: int, out_clip: str, mode: int = 0):
    frames = max(25, target_frames)
    dur = frames / FPS
    step = 0.15 / frames
    m = mode % 4

    is_valid_img = False
    if os.path.exists(img_path) and os.path.getsize(img_path) > 3000:
        try:
            with Image.open(img_path) as test_im:
                test_im.verify()
            is_valid_img = True
        except Exception:
            is_valid_img = False

    if not is_valid_img:
        safe_fallback = Image.new("RGB", (W, H), (30, 35, 45))
        safe_fallback.save(img_path, "JPEG", quality=90)

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
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        out_clip
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ==============================================================================
# PIPELINE SẢN XUẤT CHÍNH
# ==============================================================================
if st.button("⚡ Bắt Đầu Dựng Video Thành Phẩm", use_container_width=True, type="primary"):
    if not groq_key or not groq_key.strip():
        st.error("Vui lòng nhập Groq API Key!")
    elif not audio_file:
        st.error("Vui lòng tải file Voice âm thanh lên trước!")
    else:
        is_webtoon_mode = "AI Gen" in render_engine
        is_adaptive_free = "Tự do" in genre_mode

        status = st.status("Đang khởi động xưởng sản xuất video...", expanded=True)
        workdir = tempfile.mkdtemp(prefix="master_prod_")
        used_urls = set()
        used_vid_ids = set()
        char_seed = random.randint(10000, 999999)

        try:
            audio_path = os.path.join(workdir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            total_audio_dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip() or 10.0)
            total_required_frames = int(round(total_audio_dur * FPS))

            client = Groq(api_key=groq_key.strip())

            # 1. Whisper Turbo bóc tách mốc thời gian
            status.update(label="🎙️ 1/4: Whisper phân tích timestamp & bóc tách lời thoại...")
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
            detected_lang = (data.get("language") or "vietnamese").lower()
            is_english = ("en" in detected_lang)

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

                if float(seg["end"]) - cur_start >= 4.5:
                    segments.append({"start": cur_start, "end": float(seg["end"]), "text": cur_text})
                    cur_text = ""

            if cur_text:
                segments.append({"start": cur_start, "end": total_audio_dur, "text": cur_text})

            if not segments:
                segments.append({"start": 0.0, "end": total_audio_dur, "text": "story scene"})

            accumulated_frames = 0
            for i in range(len(segments)):
                if i < len(segments) - 1:
                    seg_dur = segments[i+1]["start"] - segments[i]["start"]
                    f_count = int(round(seg_dur * FPS))
                    segments[i]["target_frames"] = max(25, f_count)
                    accumulated_frames += segments[i]["target_frames"]
                else:
                    remaining = total_required_frames - accumulated_frames
                    segments[i]["target_frames"] = max(25, remaining)

            # 2. OpenAI GPT-OSS-20B Bóc tách từ khóa vật lý thực tế
            status.update(label=f"🧠 2/4: OpenAI trích xuất hành động bám sát câu thoại...")
            by_idx = {}
            batch_size = 12

            for b_start in range(0, len(segments), batch_size):
                sub_segs = segments[b_start:b_start + batch_size]
                transcript_text = "\n".join([f"[{i + b_start}] {s['text'][:90]}" for i, s in enumerate(sub_segs)])

                if is_adaptive_free:
                    prompt = f"""Bạn là một đạo diễn hình ảnh có khả năng thích ứng linh hoạt tuyệt đối.
Nhiệm vụ: Lắng nghe từng câu thoại và trích xuất ĐÚNG HÀNH ĐỘNG, ĐỊA ĐIỂM, VẬT THỂ được nói đến.

QUY TẮC CỐT LÕI:
1. KHÔNG ÉP BẤT KỲ ĐỊNH KIẾN NÀO:
   - Nói về đi uống cà phê, nhân viên nữ nhìn trộm -> 'coffee shop barista girl smiling customer table'
   - Nói về hai người bạn đi cạnh nhau, so sánh ngoại hình -> 'two young men walking street outdoor candid'
   - Nói về xin chụp ảnh, đưa điện thoại -> 'people taking selfie photo smartphone smiling outdoor'
   - Nói về xe máy, đường mưa -> 'motorcycle road heavy rain'
2. BẮT BUỘC dùng cấu trúc: [ĐỐI TƯỢNG CỤ THỂ] + [HÀNH ĐỘNG THỰC TẾ] + [ĐỊA ĐIỂM / BỐI CẢNH].
3. TUYỆT ĐỐI CẤM dùng các từ cảm xúc mơ hồ ('sad', 'depressed', 'lonely', 'thinking', 'suffering').
4. CẤM chữ viết, quote, logo.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "quán cà phê hai bạn trẻ", "query_en": "modern cafe two young men sitting table coffee"}}
]}}"""
                elif is_webtoon_mode:
                    prompt = f"""Bạn là đạo diễn hình ảnh Webtoon Manhwa Hàn Quốc. Trích xuất HÀNH ĐỘNG và BỐI CẢNH cụ thể theo lời thoại.
QUY TẮC:
1. Bóc tách hành động nhân vật đang làm (ví dụ: 'đẩy cửa bước vào lớp học', 'ngồi cạnh cửa sổ nhìn ra ngoài', 'nữ sinh lén nhìn đỏ mặt').
2. BẮT BUỘC dùng danh từ và động từ thực tế. CẤM mô tả chung chung trừu tượng.
3. Không sinh ra chữ viết hay bong bóng thoại.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "nam sinh mở cửa bước vào lớp", "query_en": "handsome boy pushing classroom door entering, students watching"}}
]}}"""
                else:
                    prompt = f"""Bạn là đạo diễn hình ảnh điện ảnh: "{genre_mode}".
Nhiệm vụ: Bóc tách VẬT THỂ và HÀNH ĐỘNG VẬT LÝ trong từng câu thoại.

QUY TẮC:
1. TUYỆT ĐỐI CẤM dùng các từ cảm xúc mơ hồ: 'sad', 'depressed', 'lonely', 'dark moody', 'thinking'.
2. BẮT BUỘC dùng công thức: [VẬT THỂ CỤ THỂ] + [HÀNH ĐỘNG VẬT LÝ] + [BỐI CẢNH].
3. Bám sát đồ vật và hành động thực tế trong câu thoại.

Đoạn thoại:
{transcript_text}

Trả về DUY NHẤT định dạng JSON:
{{"scenes": [
  {{"index": {b_start}, "query_vn": "hành động cụ thể", "query_en": "specific physical action and object"}}
]}}"""

                try:
                    llm_resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.12
                    )
                    content = llm_resp.choices[0].message.content
                    match = re.search(r'\{.*\}', content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0)).get("scenes", [])
                        for item in parsed:
                            by_idx[int(item.get("index", -1))] = item
                except Exception:
                    pass

            # 3. Dựng cảnh theo frame
            status.update(label="🎬 3/4: Đang dựng hình ảnh theo frame chính xác...", state="running")
            clips_txt = os.path.join(workdir, "clips.txt")
            with open(clips_txt, "w", encoding="utf-8") as f_clips:
                for idx, sc in enumerate(segments):
                    sc_data = by_idx.get(idx, {})
                    query_vn = sc_data.get("query_vn") or "hành động đời thực"
                    query_en = sc_data.get("query_en") or "cinematic daily action"
                    t_frames = sc["target_frames"]

                    clip_path = None
                    if is_webtoon_mode:
                        img_path = generate_manhwa_image(query_en, idx, workdir, char_seed, genre_mode)
                        clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                        create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)
                    else:
                        is_video_slot = (idx % 5 in [1, 3]) and (idx != len(segments) - 1)
                        if is_video_slot and pexels_key:
                            clip_path = fetch_broll_clip(query_en, idx, t_frames, pexels_key, workdir, used_vid_ids)

                        if not clip_path:
                            img_path = fetch_matching_image(query_vn, query_en, idx, workdir, used_urls, pexels_key, genre_mode, is_english)
                            clip_path = os.path.join(workdir, f"clip_{idx:03d}.mp4")
                            create_kenburns_clip(img_path, t_frames, clip_path, mode=idx)

                    f_clips.write(f"file '{os.path.abspath(clip_path)}'\n")

            # 4. Xuất video Master
            status.update(label="⚡ 4/4: Ghép video và nén xuất Master...", state="running")
            out_path = os.path.join(workdir, "output.mp4")

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

            status.update(label="✅ Video hoàn thành hoàn hảo!", state="complete")

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
