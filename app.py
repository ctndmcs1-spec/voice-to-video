
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

APP_TITLE = "Whiteboard AI Studio"
BATCH_SECONDS = 5 * 60
FPS = 30
WIDTH = 1280
HEIGHT = 720

# -----------------------------
# UI / configuration
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="✏️", layout="wide")

st.title("✏️ Whiteboard AI Studio")
st.caption("Voice → Groq → scenes 15–30s → 1 infographic/scene → hand-draw animation → FFmpeg → MP4")

with st.sidebar:
    st.header("🔑 API")
    groq_key = st.text_input(
        "Groq API Key",
        value=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        type="password",
    )
    pollen_key = st.text_input(
        "Pollinations API Key",
        value=st.secrets.get("POLLINATIONS_API_KEY", os.getenv("POLLINATIONS_API_KEY", "")),
        type="password",
        help="Needed for AI image generation. Groq itself does not provide text-to-image.",
    )

    st.header("🧠 Groq models")
    stt_model = st.selectbox(
        "Voice → text",
        ["whisper-large-v3-turbo", "whisper-large-v3"],
        index=0,
    )
    planner_model = st.selectbox(
        "Scene planner",
        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b", "groq/compound-mini"],
        index=0,
    )

    st.header("🎨 Image")
    image_model = st.selectbox(
        "Pollinations image model",
        ["flux", "gptimage", "seedream5", "qwen-image"],
        index=0,
    )

    st.header("🎬 Animation")
    scene_min = st.slider("Minimum scene (seconds)", 15, 25, 15)
    scene_max = st.slider("Maximum scene (seconds)", 20, 30, 30)
    if scene_max < scene_min:
        scene_max = scene_min

    draw_style = st.selectbox(
        "Animation style",
        [
            "Whiteboard + moving hand",
            "Whiteboard + moving hand + zoom",
            "Clean infographic motion",
        ],
    )

    st.header("⚙️ Safety")
    max_scenes_per_batch = st.slider("Max scenes per 5-min batch", 5, 25, 20)
    image_timeout = st.slider("Image timeout (sec)", 30, 180, 90)

# -----------------------------
# Utilities
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
        raise RuntimeError(p.stderr[-5000:] or "Command failed")
    return p.stdout

def ffprobe_duration(path):
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], timeout=60)
    return float(out.strip())

def safe_name(s, n=60):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    return (s or "scene")[:n]

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    # First try the whole response
    try:
        return json.loads(text)
    except Exception:
        pass
    # Then find the largest JSON object
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
    raise ValueError("AI did not return valid JSON")

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
    # Always process in 5-minute batches to reduce memory pressure.
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
9. The visual prompt must be in English because the image model performs better with English prompts.
10. Keep scenes between {min_s} and {max_s}; the final scene may be shorter only if the batch ends.
11. Return ONLY valid JSON.

JSON:
{{
  "scenes": [
    {{
      "start": 0.0,
      "end": 20.0,
      "title": "short Vietnamese title",
      "summary": "one sentence in Vietnamese",
      "visual_prompt": "detailed English prompt for ONE coherent whiteboard infographic image"
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
    scenes = obj.get("scenes", [])
    clean = []
    for s in scenes[:max_scenes]:
        try:
            a = max(0.0, float(s["start"]))
            b = min(batch_duration, float(s["end"]))
            if b <= a + 1:
                continue
            clean.append({
                "start": a,
                "end": b,
                "title": str(s.get("title", "Cảnh")),
                "summary": str(s.get("summary", "")),
                "visual_prompt": str(s.get("visual_prompt", "")),
            })
        except Exception:
            continue
    if not clean:
        raise ValueError("No valid scenes returned by planner")
    # Ensure coverage from 0 to batch end. Small gaps are assigned to the previous scene.
    clean[0]["start"] = 0.0
    clean[-1]["end"] = batch_duration
    return clean

def normalize_pollinations_key(api_key):
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("Thiếu POLLINATIONS_API_KEY.")
    if not (key.startswith("sk_") or key.startswith("pk_")):
        raise RuntimeError(
            "Pollinations API key không đúng định dạng hiện tại. "
            "Key hợp lệ thường bắt đầu bằng sk_ hoặc pk_."
        )
    return key


def pollinations_request(prompt, api_key, model, timeout, width=WIDTH, height=HEIGHT):
    key = normalize_pollinations_key(api_key)
    full_prompt = f"""
{prompt}

STYLE LOCK:
single coherent whiteboard infographic, 16:9 landscape composition,
pure white paper background, hand-drawn black ink line art,
simple expressive educational illustration, clean composition,
subtle red and blue accent strokes only,
clear central visual hierarchy, arrows connecting cause and effect,
leave a clean empty band near the top for a title,
NO words, NO letters, NO captions, NO watermark, NO logo,
no photorealism, no 3D render, no gradients, no clutter.
"""
    url = "https://gen.pollinations.ai/image/" + quote(full_prompt, safe="")
    params = {
        "model": model,
        "width": width,
        "height": height,
        "nologo": "true",
        "private": "true",
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            r = requests.get(
                url,
                params=params,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "image/*",
                },
                timeout=timeout,
            )
            if r.status_code == 401:
                raise RuntimeError(
                    "Pollinations trả 401 Unauthorized: API key không hợp lệ, "
                    "hết hạn hoặc chưa được cấp quyền. Hãy tạo/copy lại key tại enter.pollinations.ai."
                )
            if r.status_code == 403:
                raise RuntimeError(
                    "Pollinations trả 403 Forbidden: key không có quyền dùng model/tài nguyên này."
                )
            if r.status_code == 429:
                raise RuntimeError(
                    "Pollinations đang rate-limit (429). Hệ thống sẽ thử lại."
                )
            if r.status_code >= 400:
                detail = r.text[:700].replace("\n", " ")
                raise RuntimeError(f"Pollinations HTTP {r.status_code}: {detail}")

            content_type = r.headers.get("content-type", "").lower()
            if "image" not in content_type:
                raise RuntimeError(
                    f"Pollinations không trả ảnh (content-type={content_type}). "
                    f"Response: {r.text[:500]}"
                )
            if not r.content:
                raise RuntimeError("Pollinations trả về dữ liệu ảnh rỗng.")
            return r.content
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2 * attempt)
            else:
                raise last_error


def pollinations_image(prompt, api_key, model, output_path, timeout):
    data = pollinations_request(prompt, api_key, model, timeout)
    Path(output_path).write_bytes(data)
    try:
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            im.convert("RGB").save(output_path, quality=94)
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"File ảnh Pollinations không hợp lệ: {e}")


def test_pollinations_api(api_key, model, timeout):
    # Tiny live test: proves auth + selected image model work before a long run.
    data = pollinations_request(
        "A very simple black ink whiteboard drawing of a light bulb and a pencil, "
        "minimal composition, white background",
        api_key,
        model,
        timeout,
        width=512,
        height=288,
    )
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

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
    # White title band; keeps generated image text-free and title accurate.
    band_h = 105
    draw.rectangle([0, 0, WIDTH, band_h], fill="white")
    f = font_for(44)
    # Fit title to width
    while True:
        box = draw.textbbox((0, 0), title, font=f)
        if box[2] - box[0] <= WIDTH - 80 or getattr(f, "size", 44) <= 24:
            break
        f = font_for(max(24, getattr(f, "size", 44) - 2))
    tw = box[2] - box[0]
    draw.text(((WIDTH - tw) / 2, 25), title, fill="black", font=f)
    img.save(output_path, quality=95)

def make_hand_png(path):
    # Simple clean hand + pencil illustration. It is deliberately stylized
    # so the animation is lightweight and works without a separate asset.
    S = 260
    im = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    # palm/fingers
    d.ellipse((65, 55, 210, 220), fill=(242, 220, 190, 255), outline=(20,20,20,255), width=5)
    d.rounded_rectangle((92, 20, 145, 125), 20, fill=(242,220,190,255), outline=(20,20,20,255), width=5)
    d.rounded_rectangle((135, 35, 185, 135), 20, fill=(242,220,190,255), outline=(20,20,20,255), width=5)
    d.rounded_rectangle((48, 72, 105, 145), 22, fill=(242,220,190,255), outline=(20,20,20,255), width=5)
    # pencil
    d.polygon([(145, 185), (232, 98), (245, 111), (158, 198)], fill=(210,40,40,255), outline=(20,20,20,255))
    d.polygon([(232,98),(251,89),(245,111)], fill=(230,210,170,255), outline=(20,20,20,255))
    d.line((154, 195, 165, 207), fill=(20,20,20,255), width=5)
    im.save(path)

def render_scene(image_path, duration, output_path, hand_path, style):
    # Lightweight "draw-on" feeling:
    # - image fades in from white
    # - hand/pencil travels over the board
    # - slow camera zoom adds life
    # This is intentionally FFmpeg-only after the image is created.
    if style == "Whiteboard + moving hand + zoom":
        zoom = "zoompan=z='min(zoom+0.0007,1.10)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:fps=30:s=1280x720"
    elif style == "Clean infographic motion":
        zoom = "zoompan=z='min(zoom+0.00035,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:fps=30:s=1280x720"
    else:
        zoom = "zoompan=z='min(zoom+0.00045,1.07)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:fps=30:s=1280x720"

    # Hand is an overlay; it moves across the image while the scene plays.
    # The underlying image is whiteboard-style, so the combined result resembles
    # a hand-drawn explainer rather than a normal slideshow.
    hand_enable = style != "Clean infographic motion"
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-loop", "1", "-i", str(hand_path),
        "-t", f"{duration:.3f}",
        "-filter_complex",
        (
            f"[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
            f"pad=1280:720:(ow-iw)/2:(oh-ih)/2:white,"
            f"{zoom},fade=t=in:st=0:d=0.8[v];"
            f"[1:v]scale=170:-1[hand];"
            f"[v][hand]overlay="
            f"x='if(lt(t,{duration/2:.3f}), 80+(1180-80)*t/{duration/2:.3f}, "
            f"1180-(1180-80)*(t-{duration/2:.3f})/{duration/2:.3f})':"
            f"y='if(lt(t,{duration/2:.3f}), 500-(500-120)*t/{duration/2:.3f}, "
            f"120+(500-120)*(t-{duration/2:.3f})/{duration/2:.3f})':"
            f"enable='{str(hand_enable).lower()}'[outv]"
        ),
        "-map", "[outv]",
        "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        str(output_path),
    ]
    run_cmd(cmd, timeout=max(180, int(duration * 8)))

def render_batch(batch_audio, scenes, batch_dir, hand_path, style, progress_callback=None):
    scene_videos = []
    total = len(scenes)
    for i, s in enumerate(scenes, 1):
        img_raw = batch_dir / f"scene_{i:03d}_raw.png"
        img = batch_dir / f"scene_{i:03d}.jpg"
        vid = batch_dir / f"scene_{i:03d}.mp4"
        if not img.exists():
            pollinations_image(s["visual_prompt"], pollen_key, image_model, img_raw, image_timeout)
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
    # Match batch audio exactly.
    final_batch = batch_dir / "batch_final.mp4"
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(batch_video),
        "-i", str(batch_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
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
# Main
# -----------------------------
st.sidebar.divider()
if st.sidebar.button("🔎 TEST POLLINATIONS API", use_container_width=True):
    try:
        with st.spinner("Đang test Pollinations..."):
            test_img = test_pollinations_api(pollen_key, image_model, 60)
        st.success("Pollinations OK — API key + model hoạt động.")
        st.image(test_img, caption=f"Test model: {image_model}", use_container_width=True)
    except Exception as e:
        st.error(str(e))

audio = st.file_uploader(
    "🎤 Upload voice",
    type=["mp3", "m4a", "wav", "ogg", "webm", "mp4", "mpeg", "mpga"],
)

if audio:
    st.audio(audio)

    if st.button("🚀 CREATE VIDEO", type="primary", use_container_width=True):
        if not groq_key:
            st.error("Bạn chưa nhập GROQ_API_KEY.")
            st.stop()
        if not pollen_key:
            st.error("Bạn chưa nhập POLLINATIONS_API_KEY. Groq không có text-to-image; tool dùng Pollinations cho phần ảnh.")
            st.stop()
        try:
            pollen_key = normalize_pollinations_key(pollen_key)
        except Exception as e:
            st.error(str(e))
            st.stop()

        root = Path(tempfile.mkdtemp(prefix="wb_ai_"))
        try:
            source = root / audio.name
            source.write_bytes(audio.getbuffer())

            duration = ffprobe_duration(source)
            st.info(f"Audio: {duration/60:.2f} phút. Tool sẽ xử lý từng batch tối đa 5 phút.")

            client = groq_client(groq_key)
            batch_dir = root / "batches"
            batch_dir.mkdir()
            chunks = chunk_audio(source, batch_dir)

            hand_path = root / "hand.png"
            make_hand_png(hand_path)

            batch_videos = []
            all_scene_count = 0
            progress = st.progress(0)
            status = st.empty()

            for bi, chunk in enumerate(chunks):
                bstart = bi * BATCH_SECONDS
                bdur = ffprobe_duration(chunk)
                status.write(f"🧠 Batch {bi+1}/{len(chunks)} — transcribing...")
                tr = transcribe_file(client, chunk, stt_model)
                segs = normalize_segments(tr, bstart)
                batch_text = "\n".join(
                    f"[{x['start']:.2f}-{x['end']:.2f}] {x['text']}"
                    for x in segs
                )

                status.write(f"✂️ Batch {bi+1}/{len(chunks)} — planning scenes...")
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

                st.write(f"**Batch {bi+1}: {bdur:.1f}s → {len(scenes)} scenes**")
                for si, s in enumerate(scenes, 1):
                    st.caption(
                        f"{si:02d}. {s['start']:.1f}s–{s['end']:.1f}s — {s['title']}"
                    )

                batch_work = root / f"work_{bi+1:03d}"
                batch_work.mkdir()

                status.write(f"🎨 Batch {bi+1}/{len(chunks)} — generating images + animation...")
                def cb(frac, bi=bi):
                    progress.progress(min(1.0, (bi + frac) / len(chunks)))

                bv = render_batch(
                    chunk, scenes, batch_work, hand_path, draw_style, cb
                )

                # Move the finished batch outside the temporary work directory
                # before cleanup. Otherwise the old V1 deleted the very MP4 that
                # was needed later for final concatenation.
                saved_batch = root / f"batch_final_{bi+1:03d}.mp4"
                shutil.copy2(bv, saved_batch)
                batch_videos.append(saved_batch)
                all_scene_count += len(scenes)

                # Free scene images/videos between 5-minute batches.
                shutil.rmtree(batch_work, ignore_errors=True)

            progress.progress(1.0)
            status.write("🎬 Concatenating all batches...")

            final = root / "whiteboard_final.mp4"
            concat_batches(batch_videos, final)

            st.success(
                f"Done! {all_scene_count} scenes, processed in {len(chunks)} batch(es) of ≤5 minutes."
            )
            st.video(str(final))
            st.download_button(
                "⬇️ Download MP4",
                data=final.read_bytes(),
                file_name="whiteboard_ai_final.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)
            st.warning(
                "Nếu lỗi xảy ra ở một scene, hãy xem traceback. V1 ưu tiên dễ test và ổn định; "
                "sau khi test thành công có thể thêm resume từng scene, retry ảnh và subtitle."
            )
        finally:
            # Keep final file alive while Streamlit renders the download/video.
            # Temporary cleanup is intentionally delayed by OS/runtime.
            pass
