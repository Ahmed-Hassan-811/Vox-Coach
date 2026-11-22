# VoxCoach 🎙️

An AI voice partner for speaking practice — languages, interviews, presentations, meetings.
You talk, it talks back, and you get written feedback afterwards.

- **Brain:** Groq `openai/gpt-oss-20b`
- **Speech-to-text:** Groq `whisper-large-v3-turbo`
- **Text-to-speech:** gTTS
- **Context:** paste text or upload a CV / JD / notes / slides — BM25 retrieval when the document is large
- **Storage:** none on the server. Conversations live in the browser session and are exported by hand.

Runs on **Streamlit Community Cloud**. Previously a Gradio HuggingFace Space; moved
because free Docker/Gradio Spaces now require PRO and a paused Space cannot be restarted.

---

## Files

| File | What it is |
|---|---|
| `app.py` | The whole app — config, prompts, Groq calls, audio, document extraction, UI. Single file, no local imports. |
| `requirements.txt` | Python dependencies. `gradio` and `spaces` are gone; `streamlit>=1.40` added. |
| `README.md` | This file. |

`app.py` is laid out top to bottom: config → languages → personas/modes → the brain →
voice → context/RAG → helpers → session state → sidebar → main sections. The functions
above the `sidebar: setup` marker are pure and have no Streamlit dependency, which is
what makes them testable.

---

## Deploy

1. Push this folder to a **public** GitHub repo. Streamlit Community Cloud's free tier
   requires one — so the API key must never be committed.
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo with
   `app.py` as the entry point.
3. **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

   Paste the value with no surrounding quotes beyond the TOML ones and no trailing space.
4. The app reads the secret at first use and shows a sidebar error if it is missing, so a
   bad setup is visible immediately rather than at the first spoken sentence.

### Locally

```bash
pip install -r requirements.txt
GROQ_API_KEY=gsk_... streamlit run app.py
```

---

## What changed from the HuggingFace Space

The pipeline, prompts, personas, modes, vocabulary constraint, BM25 retrieval and
feedback are unchanged. The platform differences are:

**History no longer persists on its own.** Gradio's `gr.BrowserState` kept past
conversations in the user's own browser across reloads. Streamlit has no equivalent —
`st.session_state` is server-side and dies with the session or when the app sleeps.
History now has explicit **Download** and **Restore** buttons and says plainly that a
reload clears it. Nothing is stored server-side either way, so the privacy promise is
intact; the convenience is what was lost.

**Hold-SPACE-to-record is gone.** It injected JavaScript that reached into Gradio's DOM
to click the record button. Streamlit renders components in sandboxed iframes, so a
component cannot reach the parent page. Click the mic widget instead.

**The `@spaces.GPU` stub is gone**, along with the `spaces` dependency. It existed only
to satisfy a ZeroGPU startup check on HF; all inference is Groq's.

**No temp files.** The Space wrote each reply with `tempfile.mktemp()` — deprecated,
racy, and leaking one `.mp3` per reply for the life of the process. gTTS now writes to
an in-memory buffer and `st.audio` takes the bytes.

**A rerun guard on the mic.** Streamlit reruns the whole script on every interaction and
`st.audio_input` returns the same recording until the user records again. Without a
guard, one spoken sentence is transcribed and answered on every rerun — a real Whisper
call and a real chat call each time. A SHA-256 fingerprint of the audio bytes makes a
recording process exactly once.

**Document extraction was rewritten after a real deck failed.** The Space read
`.pptx` as `shape.text for shape in slide.shapes if shape.has_text_frame`, which
quietly loses most real decks: `slide.shapes` does not recurse into **groups**, and a
group is not a text frame — so a deck built the way designers build them extracted to
nothing. **Tables** and **speaker notes** were dropped for the same reason. All three
are read now, each shape inside its own `try`, so one unreadable object cannot cost the
whole file. `extract_text` also returns `(text, error)` so the app can tell *"could not
open this"* from *"opened it, there is no text in it"* — a scanned PDF or an
all-images deck is not a broken file, and saying so sends people off re-exporting
something that was fine.

**Picture decks can be read with a vision model.** Slides exported as images —
NotebookLM, Canva, "export as images" — contain no text at all, so no extractor can
help. When an uploaded `.pptx` opens cleanly but has no text and does have images, the
sidebar offers a **Read the slides with AI** button, with the slide count and a rough
time stated. It sends one downscaled slide image per call to a Groq vision model and
asks for a transcription, not a description. It is never automatic, because it costs a
call per slide.

Details worth knowing before changing it:

- The model is picked from a **fallback list** (`VISION_MODELS`), not hard-coded.
  Groq's vision line-up moves — Maverick was shut down in March 2026 — and availability
  differs per account, so a 404 on one model must not end the job. Whichever answers
  first is remembered for the rest of the deck.
- Free-tier vision is **30 requests/minute**, which an 18-slide deck can brush, so a
  429 backs off and retries rather than losing the slide.
- Images are **downscaled to 1600px** (image tokens scale with resolution) and then
  encoded as **whichever of PNG or JPEG is smaller** — a slide that is mostly flat
  colour and text is smaller as PNG, and always choosing JPEG made those slides bigger.
  Groq caps base64 images at 4 MB; quality steps down before a slide is given up on.
- **Partial results are kept.** If slide 12 fails, the other 17 are still returned and
  the failures are named.
- There is a **60-slide cap** so a large deck cannot quietly spend hundreds of calls.

**The section radio has no `key=`.** With one, Streamlit refuses any later assignment to
that state key in the same run (`StreamlitWidgetAlreadyInstantiatedError`), and
"End & Save conversation" does exactly that when it jumps to Feedback — so the app would
have crashed immediately after the feedback call was paid for. It is driven by `index`
instead. There is a regression test for this.

---

## Known trade-offs on the free tier

- The app **sleeps after ~12 hours** without traffic and cold-starts on the next visit.
  There is no paid tier that disables this.
- The repo must be **public**; the key lives only in Streamlit secrets.
- Sleeping also ends every session, which is why History export exists.