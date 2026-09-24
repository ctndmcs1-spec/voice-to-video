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

VERSION = '11.0'

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
    """Do not mistake an API quota failure for malformed scene JSON."""


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
            message = str(exc)
            if 'request too large' in message.lower():
                limit = re.search(r'\blimit\s*:?\s*([\d,]+)', message, re.I)
                detail = f" Hạn mức API báo: {limit.group(1)}." if limit else ''
                raise PlannerRateLimit(
                    f"Yêu cầu vượt hạn mức Groq cho {model} (ngân sách đầu ra hiện tại: {max_tokens})."
                    + detail + " Giảm 'Token đầu ra tối đa' để tool chia đợt nhỏ hơn; kiểm tra Limits của tài khoản. Chờ rồi gửi nguyên yêu cầu sẽ không giải quyết lỗi này."
                ) from exc
            if attempt == 2:
                raise PlannerRateLimit("Groq vẫn giới hạn 429 sau các lần chờ. Tiến độ đã lưu; thử tiếp sau theo Limits của tài khoản.") from exc
            response = getattr(exc, 'response', None)
            headers = getattr(response, 'headers', {}) or {}
            retry = headers.get('retry-after')
            delay = min(30.0, 5.0 * (2**attempt))
            if retry is not None:
                try: delay = max(0.0, float(retry))
                except (TypeError, ValueError):
                    try: delay = max(0.0, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError): pass
            delay += 0.5
            if not math.isfinite(delay) or waited + delay > 60:
                raise PlannerRateLimit("Groq yêu cầu chờ lâu hơn 60 giây. Tiến độ đã lưu; hãy thử tiếp sau, không cần tạo lại cảnh thành công.") from exc
            notify(f"Groq đang giới hạn tốc độ. Chờ {delay:.1f}s rồi thử lại ({attempt+2}/3).")
            sleep(delay); waited += delay
