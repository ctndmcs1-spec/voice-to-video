# Whiteboard AI Studio V1.1

Voice → Groq Whisper → semantic scenes 15–30s → one coherent infographic/scene → Pollinations image → hand-draw style animation → FFmpeg → MP4.

## V1.1 changes
- Pollinations API key validation (`sk_` / `pk_`).
- New **TEST POLLINATIONS API** button for a small live image test before creating a long video.
- Better Pollinations errors for 401 / 403 / 429 / non-image responses.
- Automatic retry up to 3 times for temporary image-generation failures.
- Fixes a V1 cleanup bug: completed 5-minute batch MP4s are copied outside the work folder before cleanup, so final concatenation can find them.

## Streamlit secrets
Use `.streamlit/secrets.toml` (do not commit it):

```toml
GROQ_API_KEY = "gsk_..."
POLLINATIONS_API_KEY = "sk_..."
```

A Pollinations secret server-side key normally starts with `sk_`. The API also supports `pk_` keys, but secret keys are intended for server-side apps.

## Test order
1. Deploy to Streamlit.
2. Put the Pollinations key in the sidebar (or Streamlit Secrets).
3. Choose an image model.
4. Tap **TEST POLLINATIONS API**.
5. Only after it says **Pollinations OK**, upload a short 30–60 second voice and press **CREATE VIDEO**.

Start small before trying a full 5-minute batch.
