"""Small, UI-independent persistence and media helpers for Studio V11."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

VERSION = '11.2'

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()

def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary: temporary.unlink(missing_ok=True)

def artifact_ok(path, key):
    path = Path(path)
    meta = read_json(str(path) + '.json', {})
    return (isinstance(meta, dict) and path.is_file() and path.stat().st_size > 0
            and meta.get('key') == key and meta.get('sha256') == file_digest(path))

def mark_artifact(path, key):
    write_json(str(path) + '.json', {'key': key, 'sha256': file_digest(path)})

@contextmanager
def job_lock(root):
    """OS releases the lock even when a Linux worker crashes."""
    import fcntl
    with open(Path(root) / 'job.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Dự án đang được xử lý. Đợi lượt đang chạy hoàn tất.')
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)

class RawVideoWriter:
    """Drain encoder errors to disk, close/kill reliably, publish only complete MP4."""
    def __init__(self, cmd):
        self.output = Path(cmd[-1])
        self.part = self.output.with_name(self.output.stem + '.part.mp4')
        self.cmd = cmd[:-1] + [str(self.part)]

    def __enter__(self):
        self.errors = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=self.errors)
        except BaseException:
            self.errors.close(); raise
        self.stdin = self.proc.stdin
        return self

    def __exit__(self, typ, value, tb):
        failure = None
        try:
            try: self.stdin.close()
            except BrokenPipeError: pass
            if typ is not None:
                self.proc.kill()
            try: self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(); failure = 'FFmpeg hết thời gian chờ.'
            self.errors.seek(0)
            details = self.errors.read()[-5000:].decode(errors='replace')
            if self.proc.returncode:
                failure = details or 'FFmpeg không xuất được video.'
            if typ is None and failure is None:
                os.replace(self.part, self.output)
        finally:
            self.errors.close()
            self.part.unlink(missing_ok=True)
        if typ is None and failure:
            raise RuntimeError(failure)
        return False

def frame_durations(scenes, fps):
    """Round cumulative boundaries rather than accumulating one error per scene."""
    counts = [round(s['end'] * fps) - round(s['start'] * fps) for s in scenes]
    if any(n < 1 for n in counts):
        raise ValueError('Có cảnh ngắn hơn một khung hình; hãy sửa kịch bản.')
    return [n / fps for n in counts]

def script_slice(text, start, end, duration):
    words = text.split()
    if not words or duration <= 0: return ''
    a = round(len(words) * max(0, start) / duration)
    b = round(len(words) * min(duration, end) / duration)
    return ' '.join(words[a:b])

def srt_text(segments):
    def stamp(seconds):
        ms = max(0, round(seconds * 1000))
        h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f'{h:02}:{m:02}:{s:02},{ms:03}'
    cues = []; last_end = 0.0
    for segment in segments:
        start, end = float(segment['start']), float(segment['end'])
        text = ' '.join(str(segment['text']).split())
        if not text or not math.isfinite(start) or not math.isfinite(end): continue
        lines = textwrap.wrap(text, width=42, break_long_words=False)
        chunks = ['\n'.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
        start = max(last_end, start)
        if end <= start: continue
        span = (end - start) / len(chunks)
        for i, chunk in enumerate(chunks):
            cues.append(f'{len(cues)+1}\n{stamp(start+i*span)} --> {stamp(start+(i+1)*span)}\n{chunk}\n')
        last_end = end
    return '\n'.join(cues)

class PlannerRateLimit(RuntimeError):
    """Keep the provider's evidence separate from the user-facing explanation."""
    def __init__(self, message, details=""):
        super().__init__(message)
        self.details = details


def rate_limit_details(error):
    import re
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        payload = body.get("error", body)
        message = payload.get("message") if isinstance(payload, dict) else None
    else:
        message = None
    message = str(message or error)
    message = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", message)
    number = r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)"
    values = {}
    for field in ("limit", "used", "requested"):
        match = re.search(r"\b" + field + r"\s*:?\s*" + number, message, re.I)
        values[field] = float(match.group(1).replace(",", "")) if match else None
    lower = message.lower()
    values["axis"] = ("OTPM" if "otpm" in lower or "output tokens per minute" in lower else
                      "ITPM" if "itpm" in lower or "input tokens per minute" in lower else
                      "TPM" if "tokens per minute" in lower or "tpm" in lower else "unknown")
    values["message"] = message
    return values


def planner_capacity(output_budget):
    # Conservative planning estimate, not a provider's published quota.
    return max(1, (int(output_budget) - 512) // 650)


def chat_completion(client, model, messages, max_tokens, notify=lambda message: None,
                    sleep=None):
    import re
    import time
    from email.utils import parsedate_to_datetime
    if sleep is None: sleep = time.sleep
    waited = 0.0
    for attempt in range(3):
        try:
            kwargs = {'model': model, 'messages': messages, 'temperature': 0.15, 'max_tokens': max_tokens}
            if model == 'qwen/qwen3.8-27b': kwargs['reasoning_effort'] = 'none'
            elif model in ('openai/gpt-oss-120b', 'openai/gpt-oss-20b'): kwargs['reasoning_effort'] = 'low'
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 429: raise
            detail = rate_limit_details(exc)
            message = detail["message"]
            limit = detail["limit"]; requested = detail["requested"]
            oversized = (limit is not None and requested is not None and requested > limit)
            oversized = oversized or (detail["axis"] == "OTPM" and limit is not None and max_tokens > limit)
            if oversized or 'request too large' in message.lower():
                figures = f" Limit={limit:g}." if limit is not None else ""
                if requested is not None: figures += f" Requested={requested:g}."
                explanation = (
                    "Một yêu cầu vượt toàn bộ hạn mức của loại token được báo; cần giảm kích thước yêu cầu."
                    if oversized else
                    "Groq báo Request too large; cần đối chiếu Limit/Requested và loại hạn mức trong chi tiết gốc."
                )
                if detail["axis"] == "OTPM":
                    explanation += " Giảm 'Token đầu ra tối đa / yêu cầu' xuống dưới Limit; tool sẽ chia đợt nhỏ hơn."
                elif detail["axis"] in ("TPM", "ITPM"):
                    explanation += " Cần giảm cả văn bản đầu vào/số cảnh, không chỉ token đầu ra."
                raise PlannerRateLimit(
                    f"Groq {model}: {detail['axis']}, ngân sách đầu ra ứng dụng={max_tokens}." + figures + " " + explanation,
                    details=message,
                ) from exc
            if attempt == 2:
                raise PlannerRateLimit("Groq vẫn giới hạn 429 sau các lần chờ. Tiến độ đã lưu; thử tiếp sau theo Limits của tài khoản.", details=message) from exc
            response = getattr(exc, 'response', None)
            headers = getattr(response, 'headers', {}) or {}
            retry = headers.get('retry-after')
            delay = min(30.0, 5.0 * (2**attempt))
            if retry is not None:
                try: delay = max(0.0, float(retry))
                except (TypeError, ValueError):
                    try: delay = max(0.0, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError): pass
            if retry is None:
                match = re.search(r"try again in\s+(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?", message, re.I)
                if match and any(match.groups()):
                    delay = float(match.group(1) or 0) * 60 + float(match.group(2) or 0)
            delay += 0.5
            if not math.isfinite(delay) or waited + delay > 300:
                raise PlannerRateLimit("Groq yêu cầu chờ vượt ngân sách tự đợi 5 phút. Tiến độ đã lưu; hãy thử tiếp sau, không cần tạo lại cảnh thành công.", details=message) from exc
            notify(f"Groq đang giới hạn tốc độ. Chờ {delay:.1f}s rồi thử lại ({attempt+2}/3).")
            remaining = delay
            while remaining > 0:
                interval = min(30.0, remaining)
                sleep(interval); remaining -= interval
                if remaining > 0: notify(f"Đang chờ hạn mức Groq: còn khoảng {remaining:.0f}s.")
            waited += delay

def align_script_to_segments(script, segments, minimum_score=0.55):
    """Correct segment text by monotonic word alignment, retaining STT times.

    This is text alignment against recognized words, not acoustic forced alignment.
    Large unmatched spans are rejected instead of being stretched across the voice.
    """
    import difflib
    import re
    import unicodedata
    def normalize(word):
        word = unicodedata.normalize('NFD', word.casefold().replace('đ', 'd'))
        return ''.join(c for c in word if not unicodedata.combining(c) and c.isalnum())
    target = [(word, normalize(word)) for word in script.split()]
    target = [(word, normal) for word, normal in target if normal]
    heard = []
    for index, segment in enumerate(segments):
        for word in str(segment.get('text', '')).split():
            normal = normalize(word)
            if normal: heard.append((word, normal, index))
    if not target or not heard:
        raise ValueError('Không đủ text hoặc lời nhận dạng để căn khớp. Hãy kiểm tra voice và kịch bản.')
    matcher = difflib.SequenceMatcher(None, [word[1] for word in heard],
                                     [word[1] for word in target], autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    score = matches / max(len(heard), len(target))
    if score < minimum_score:
        raise ValueError(f'Text và voice chỉ khớp {score:.0%} theo từ. Dừng để tránh ghép sai; hãy dùng đúng bản text đã đọc hoặc kiểm tra nhận dạng voice.')
    corrected = [[] for _ in segments]
    uncertain = set(); omitted = []
    for operation, a, b, x, y in matcher.get_opcodes():
        if operation == 'equal':
            for h, t in zip(heard[a:b], target[x:y]):
                corrected[h[2]].append(t[0])
        elif operation == 'replace':
            if max(b-a, y-x) > 12:
                raise ValueError('Có đoạn text khác voice quá dài để tự căn an toàn. Hãy sửa text đúng lời đọc rồi thử lại.')
            # Small ASR substitutions inherit their original segment boundaries.
            for j, word in enumerate(target[x:y]):
                source_index = a + min(b-a-1, j*(b-a)//max(1,y-x))
                corrected[heard[source_index][2]].append(word[0])
                uncertain.add(heard[source_index][2])
        elif operation == 'delete':
            # Spoken ad-lib absent from the script must remain in the timeline.
            for word in heard[a:b]:
                corrected[word[2]].append(word[0]); uncertain.add(word[2])
        else:
            # Script-only words have no measured time. Do not manufacture one.
            omitted.extend(word[0] for word in target[x:y])
            if y-x > 12:
                raise ValueError('Text có đoạn dài không tìm thấy trong voice. Không thể tự gán thời gian cho đoạn này.')
    output = []
    for index, segment in enumerate(segments):
        text = ' '.join(corrected[index])
        if not text:
            text = str(segment.get('text','')).strip(); uncertain.add(index)
        output.append(dict(segment, text=text, original_text=str(segment.get('text','')),
                           alignment_review=index in uncertain))
    return output, {'score':score, 'review_segments':len(uncertain),
                    'script_only_words':' '.join(omitted)}


def scene_image_key(scene, arrows=True, shadow=True, style='comic', language='vi'):
    return fingerprint({'version': VERSION, 'scene': scene, 'arrows': arrows,
                        'shadow': shadow, 'style': style, 'language': language})


def scene_narration(scene, segments):
    return ' '.join(str(s['text']).strip() for s in segments
                    if float(s['end']) > float(scene['start']) and float(s['start']) < float(scene['end']))
