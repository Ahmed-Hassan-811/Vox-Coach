"""
VoxCoach — a voice speaking-practice partner (single-file Streamlit app).

Ported from the Gradio HuggingFace Space, 22 Nov 2026.

Pipeline is unchanged:
    mic -> Groq Whisper (STT) -> Groq gpt-oss-20b (brain) -> gTTS (TTS)
Context is unchanged:
    paste text OR upload a document -> extract -> BM25 retrieval when large

WHAT CHANGED IN THE PORT — read this before editing
---------------------------------------------------
1. HISTORY NO LONGER PERSISTS BY ITSELF.
   Gradio's gr.BrowserState stored past conversations in the user's own
   browser. Streamlit has no equivalent: st.session_state lives on the
   server and dies when the session ends or the app sleeps. Rather than
   pretend otherwise, History now has explicit Download / Restore buttons
   and says plainly that a reload clears it. Nothing is written to a
   server-side database, so the privacy promise is intact — the
   convenience is what was lost.

2. HOLD-SPACE-TO-RECORD IS GONE.
   It worked by injecting JS that reached into Gradio's DOM to click the
   record button. Streamlit renders custom components inside sandboxed
   iframes, so a component cannot reach the parent page to do that.
   Click the mic widget instead.

3. THE @spaces.GPU STUB IS GONE, with the `spaces` dependency. It existed
   only to satisfy a ZeroGPU startup check on HF. All inference is Groq's.

4. NO TEMP FILES. The Space wrote each TTS clip with tempfile.mktemp(),
   which is deprecated, racy, and leaked an .mp3 per reply for the life of
   the process. gTTS writes to an in-memory buffer now and st.audio takes
   the bytes directly.

5. RERUN GUARD ON THE MIC — the important one.
   Streamlit reruns this whole script on every interaction, and
   st.audio_input keeps returning the SAME recording until it is cleared.
   Without a guard, one spoken sentence is transcribed and answered again
   on every rerun — billed every time. _audio_fingerprint() + the
   last_audio_id check make a recording process exactly once.
"""

import base64
import hashlib
import html
import io
import json
import os
import re
import time
import zipfile

import streamlit as st
from groq import Groq
from gtts import gTTS
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------- config
CHAT_MODEL = "openai/gpt-oss-20b"
STT_MODEL = "whisper-large-v3-turbo"
SMALL_DOC = 6000          # chars: below this use the whole doc, above this retrieve
HISTORY_SCHEMA = "voxcoach_history_v1"   # carried into exports so a restore can check it

st.set_page_config(page_title="VoxCoach", page_icon="🎙️", layout="wide")


@st.cache_resource
def get_client():
    """One Groq client for the whole server process.

    st.secrets first (how Streamlit Community Cloud supplies it), then the
    environment so `GROQ_API_KEY=... streamlit run app.py` works locally.
    cache_resource keeps this from rebuilding on every rerun.
    """
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        # No secrets.toml at all — local run. Fall through to the env var.
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key)


# ---------------------------------------------------------------- languages
# (display name -> short code used by gTTS and Whisper). Display name goes to the LLM.
LANGUAGES = [
    ("English (US)", "en"), ("English (UK)", "en"), ("Urdu", "ur"), ("Hindi", "hi"),
    ("Arabic", "ar"), ("Spanish (Spain)", "es"), ("Spanish (Latin Am.)", "es"), ("French", "fr"),
    ("German", "de"), ("Italian", "it"), ("Portuguese (Brazil)", "pt"), ("Chinese (Mandarin)", "zh-CN"),
    ("Japanese", "ja"), ("Korean", "ko"), ("Russian", "ru"), ("Turkish", "tr"), ("Bengali", "bn"),
]
LANG_NAMES = [n for n, _ in LANGUAGES]
CODE = dict(LANGUAGES)                         # name -> short code

# ---------------------------------------------------------------- personas / modes
PERSONAS = {
    "Default": "",
    "Friendly Tutor": "You are warm, encouraging and patient, like a kind teacher.",
    "Strict Interviewer": "You are a no-nonsense professional interviewer who probes for detail.",
    "Entrepreneur": "You are a busy startup founder who talks about business and growth.",
    "Psychologist": "You are a calm, thoughtful psychologist who asks reflective questions.",
    "Casual Native": "You are a relaxed native speaker chatting like a friend.",
    "Skeptical Investor": "You are a sharp investor who challenges every claim.",
}
PERSONA_NAMES = list(PERSONAS.keys())

MODE_PROMPTS = {
    "Language": "Have a casual, friendly conversation to help the user practise speaking. "
                "Gently model correct phrasing by example; do not stop to correct every error.",
    "Interview": "Act as a professional interviewer. Ask realistic questions ONE at a time. "
                 "Briefly react to each answer, then ask the next question.",
    "Presentation": "Act as a supportive audience and coach. Let the user present, then ask the "
                    "kind of questions a real audience would ask.",
    "Meeting": "Act as a colleague in a work meeting. Discuss the topic naturally, push back "
               "politely, and keep it professional.",
}


# ---------------------------------------------------------------- the brain (engine)
# Unchanged from the Space. These are pure functions of their arguments, which
# is what makes them testable without a Streamlit runtime or a Groq key.
def build_system_prompt(mode, persona, target_lang, native_lang, vocabulary, doc_context):
    p = ["You are a friendly, patient speaking-practice partner. Hold a natural spoken "
         "back-and-forth. Keep EVERY reply SHORT (1-3 sentences) because it is read aloud. "
         "Never lecture. Ask ONE question at a time.\n"
         "ALWAYS react to what the user just said BEFORE adding anything new. Do NOT repeat a "
         "sentence or question you already used, and NEVER ask something already answered -- move "
         "forward with a new small question each turn. You may echo back names, numbers and "
         "places the user mentions even if they are not in the target words."]
    p.append(f"Conduct this entire session in {target_lang}.")
    p.append(MODE_PROMPTS.get(mode, MODE_PROMPTS["Language"]))
    if PERSONAS.get(persona):
        p.append(PERSONAS[persona])
    if mode == "Language":
        p.append(f"When you use a new or difficult word, add a very short explanation in "
                 f"{native_lang} in brackets right after it.")
    if vocabulary and vocabulary.strip():
        p.append("STRICT VOCABULARY CONSTRAINT: Speak using ONLY these target words plus the "
                 "simplest glue words (a, the, is, are, have, do, go, you, I, and, to). Do NOT use "
                 "advanced vocabulary. If you must introduce a new word, explain it in one very "
                 f"simple phrase in {native_lang}. Keep sentences very short. Keep finding NEW "
                 "things to say -- never loop on the same greeting or question.\n"
                 f"TARGET WORDS: {vocabulary.strip()}")
    if doc_context and doc_context.strip():
        p.append("Use this user-provided context to make the session realistic (e.g. base "
                 "interview questions on the CV and job description):\n\n" + doc_context.strip()[:6000])
    return "\n\n".join(p)


def clean_messages(msgs):
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def _complete(messages, temperature=0.7, max_tokens=1024, freq=0.0, pres=0.0):
    """gpt-oss reasons before answering: reasoning_effort='low' + max_completion_tokens keep it
    from returning empty content; penalties cut repetition on the chat turn.

    Kept exactly as the Space had it — this combination was tuned against the
    model's empty-content failure and is load-bearing, not incidental."""
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not set. Add it in Streamlit secrets.")
    r = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=temperature,
        max_completion_tokens=max_tokens, reasoning_effort="low",
        frequency_penalty=freq, presence_penalty=pres,
    )
    return r.choices[0].message.content or ""


def chat_logic(messages, mode, persona, target_lang, native_lang, vocabulary, doc_context):
    sys = build_system_prompt(mode, persona, target_lang, native_lang, vocabulary, doc_context)
    return _complete([{"role": "system", "content": sys}] + clean_messages(messages),
                     temperature=0.8, freq=0.6, pres=0.5)


def feedback_logic(transcript, feedback_lang):
    prompt = ("You are a supportive speaking coach. Give SANDWICH feedback on the transcript:\n"
              "1) 2 specific strengths. 2) 2-3 concrete improvements (grammar/vocab/fluency/clarity) "
              "each with a corrected example. 3) One encouraging line + one thing to try next.\n"
              f"Under 200 words, warm and plain. Write it in {feedback_lang}.\n\nTRANSCRIPT:\n" + transcript)
    return _complete([{"role": "user", "content": prompt}], temperature=0.5)


def translate_logic(text, target_lang):
    prompt = (f"Translate the following into {target_lang}. Keep meaning and friendly tone. "
              "Output ONLY the translation.\n\n" + text)
    return _complete([{"role": "user", "content": prompt}], temperature=0.2)


# ---------------------------------------------------------------- voice
def transcribe(audio_bytes, filename, stt_code):
    """Groq Whisper. Takes bytes rather than a path — st.audio_input hands us
    an in-memory UploadedFile, and writing it to disk first would buy nothing."""
    if not audio_bytes:
        return ""
    try:
        client = get_client()
        if client is None:
            return ""
        tr = client.audio.transcriptions.create(
            file=(filename or "speech.wav", audio_bytes),
            model=STT_MODEL,
            language=stt_code.split("-")[0],   # bare ISO code, e.g. "ja"
        )
        return (tr.text or "").strip()
    except Exception as e:
        print("STT error:", e)
        return ""


def speak(text, tts_code):
    """gTTS straight into memory. The Space used tempfile.mktemp() and left
    one .mp3 on disk per reply; st.audio accepts bytes, so there is no reason
    to touch the filesystem at all."""
    if not text or not text.strip():
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=tts_code).write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        print("TTS error:", e)   # unsupported language -> stay silent, text still shows
        return None


def _audio_fingerprint(data):
    """Identity for a recording, so the same clip is never processed twice.

    st.audio_input returns the SAME UploadedFile on every rerun until the user
    records again. Streamlit reruns on any widget change, so without this the
    app would re-transcribe and re-answer one sentence repeatedly — each pass
    a real Whisper call and a real chat call."""
    return hashlib.sha256(data).hexdigest() if data else ""


# ---------------------------------------------------------------- context + lightweight RAG
def _pptx_text(raw):
    """Pull text out of a .pptx.

    The first version of this was `sh.text for sh in slide.shapes if
    sh.has_text_frame`, which looked reasonable and lost most real decks:

      * GROUPED SHAPES were skipped entirely. slide.shapes does not recurse,
        and a group is not itself a text frame — so a deck built the way
        designers actually build them (everything grouped) extracted to
        nothing at all, and the app reported "could not read that file".
      * TABLES were skipped. A table is not a text frame; its text lives in
        table.rows[].cells[].
      * SPEAKER NOTES were skipped, and in a talk deck the notes are often
        where the actual sentences are.

    Each shape is also read inside its own try/except, so one unreadable
    object cannot cost the whole file — losing one chart title is a fine
    trade for keeping the other ninety slides."""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out = []

    def walk(shapes, depth=0):
        if depth > 10:          # groups can nest; this bounds the recursion
            return
        for sh in shapes:
            try:
                if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                    walk(sh.shapes, depth + 1)
                    continue
            except Exception:
                pass
            try:
                if getattr(sh, "has_text_frame", False):
                    t = sh.text_frame.text
                    if t and t.strip():
                        out.append(t.strip())
                if getattr(sh, "has_table", False):
                    for row in sh.table.rows:
                        cells = [(c.text or "").strip() for c in row.cells]
                        if any(cells):
                            out.append(" | ".join(cells))
                if getattr(sh, "has_chart", False) and sh.chart.has_title:
                    out.append(sh.chart.chart_title.text_frame.text.strip())
            except Exception:
                continue

    prs = Presentation(io.BytesIO(raw))
    for slide in prs.slides:
        walk(slide.shapes)
        try:
            if slide.has_notes_slide:
                n = slide.notes_slide.notes_text_frame.text
                if n and n.strip():
                    out.append(n.strip())
        except Exception:
            pass

    text = "\n".join(out)
    if text.strip():
        return text

    # Fallback: read the slide XML directly.
    #
    # python-pptx only exposes the shape kinds it models. SmartArt and some
    # other graphic frames carry their text in parts of the XML it will not
    # hand over, so a deck can be full of words and still extract to nothing
    # through the shapes API. Every text run in an Office file — whatever
    # drew it — is an <a:t> element, so scraping those finds anything the
    # typed API missed. It loses the structure, which is why it is a
    # fallback and not the first attempt.
    return _pptx_raw_xml_text(raw)


def _pptx_raw_xml_text(raw):
    """Every <a:t> run in every slide and notes part, in slide order."""
    out = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            parts = sorted(
                n for n in z.namelist()
                if re.match(r"ppt/(slides|notesSlides)/[^/]+\.xml$", n)
            )
            for p in parts:
                try:
                    xml = z.read(p).decode("utf-8", errors="ignore")
                except Exception:
                    continue
                runs = re.findall(r"<a:t>(.*?)</a:t>", xml, re.S)
                line = " ".join(html.unescape(r) for r in runs).strip()
                if line:
                    out.append(line)
    except Exception as e:
        print("raw pptx scrape failed:", e)
    return "\n".join(out)


def _pptx_inventory(raw):
    """(slide count, embedded image count) straight from the package.

    Both numbers, not just images: a .pptx stores one copy of a repeated
    image and references it from many slides, so image count alone
    under-reports a deck that reuses a background or a logo. "40 slides,
    3 images" and "40 slides, 40 images" describe very different files, and
    the second is the one that means every slide is a picture."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            slides = sum(1 for n in names
                         if re.match(r"ppt/slides/slide\d+\.xml$", n))
            images = sum(1 for n in names if n.startswith("ppt/media/"))
            return slides, images
    except Exception:
        return 0, 0


# ---------------------------------------------------------------- reading picture slides
#
# Decks exported as one image per slide — NotebookLM, Canva, "export as
# images" — contain no text at all, so no extractor can help. The only way
# to read them is to look at them, which means a vision model.
#
# This costs one API call per slide and is therefore NEVER automatic. The
# sidebar offers it as a button, with the slide count shown, after an upload
# has already been found to be text-free.

# Tried in order; the first that answers is remembered for the rest of the
# deck. A single hard-coded model is the wrong shape here: Groq's vision
# line-up moves (Maverick was shut down in March 2026) and availability
# differs per account, so a 404 on one model must not end the job. The last
# entry is a best guess and is expected to 404 on most accounts — harmless,
# because the chain stops at whichever one works.
VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3.6-27b",
]

VISION_PROMPT = (
    "Transcribe every word of text visible on this slide, exactly as written. "
    "Include the title, all bullet points, table cells, labels inside diagrams, "
    "and axis labels on charts. Preserve the reading order. "
    "Do not summarise, explain, describe the images, or add any commentary. "
    "If the slide contains no readable text at all, reply with exactly: NO_TEXT"
)

# Groq accepts base64 images up to 4 MB. Slides are wide, and 1600px on the
# long edge keeps body text legible while landing far under that — a 23 MB
# deck of PNG screenshots becomes a few hundred KB per slide, which is the
# difference between this taking half a minute and several minutes.
VISION_MAX_PX = 1600
VISION_MAX_B64 = 3_500_000          # headroom under Groq's 4 MB base64 cap
VISION_SLIDE_CAP = 60               # refuse to spend hundreds of calls silently


def _slide_images(raw):
    """One image blob per slide, in slide order.

    Takes the FIRST picture on each slide: in an exported deck that is the
    slide itself. Reading every picture would re-send logos and decorations
    as if they were content, at a call apiece."""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    out = []

    def first_picture(shapes, depth=0):
        if depth > 10:
            return None
        for sh in shapes:
            try:
                if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                    found = first_picture(sh.shapes, depth + 1)
                    if found:
                        return found
                    continue
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    return sh.image.blob
            except Exception:
                continue
        return None

    for slide in Presentation(io.BytesIO(raw)).slides:
        out.append(first_picture(slide.shapes))
    return out


def _to_data_url(blob, max_px=VISION_MAX_PX, quality=82):
    """Downscale a slide image and encode it small enough for the API.

    Downscaling always happens: image tokens scale with resolution, so it
    cuts the bill whatever the file size does.

    The ENCODING is then chosen by measurement, not assumption. JPEG is the
    obvious pick for a photographic slide, but a slide that is mostly flat
    colour and text — which plenty are — is smaller as PNG, and an earlier
    version of this function silently made those slides bigger. Both are
    encoded and the smaller one is sent.

    If neither fits the cap, JPEG quality is stepped down before giving up,
    rather than sending a request certain to be rejected."""
    from PIL import Image

    im = Image.open(io.BytesIO(blob))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.thumbnail((max_px, max_px))

    def encoded(fmt, **kw):
        buf = io.BytesIO()
        im.save(buf, fmt, **kw)
        return buf.getvalue()

    png = encoded("PNG", optimize=True)
    jpg = encoded("JPEG", quality=quality, optimize=True)

    best, mime = (png, "image/png") if len(png) <= len(jpg) else (jpg, "image/jpeg")
    candidates = [(best, mime)]
    # Progressively cheaper fallbacks, only consulted if the best is too big.
    for q in (60, 40):
        candidates.append((encoded("JPEG", quality=q, optimize=True), "image/jpeg"))

    for data, mt in candidates:
        b64 = base64.b64encode(data).decode("ascii")
        if len(b64) <= VISION_MAX_B64:
            return f"data:{mt};base64,{b64}"
    return None                     # genuinely too big; caller skips the slide


def _vision_read(data_url, state):
    """One slide through the vision model.

    `state` carries the model that last worked and is mutated in place, so a
    deck does not re-discover the working model — or re-pay for the 404s —
    on every slide."""
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not set.")

    order = ([state["model"]] if state.get("model")
             else list(VISION_MODELS))
    last_err = None

    for model in order:
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]}],
                    temperature=0,          # transcription, not writing
                    max_completion_tokens=1500,
                )
                state["model"] = model
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                msg = str(e)
                last_err = e
                # Free tier is 30 requests/minute; an 18-slide deck can brush
                # that. Back off and retry rather than losing the slide.
                if "429" in msg or "rate" in msg.lower():
                    time.sleep(2 * (attempt + 1))
                    continue
                break                       # 404 / auth / bad request: next model
    raise last_err if last_err else RuntimeError("No vision model available.")


def read_slides_with_vision(raw, progress=None):
    """Transcribe a picture deck. Returns (text, errors[]).

    Partial results are kept: if slide 12 fails, the other 17 are still
    worth having, and the caller is told which ones were lost."""
    blobs = _slide_images(raw)
    if len(blobs) > VISION_SLIDE_CAP:
        return "", [f"deck has {len(blobs)} slides, over the {VISION_SLIDE_CAP}-slide limit"]

    state, parts, errors = {"model": None}, [], []

    for i, blob in enumerate(blobs, start=1):
        if progress:
            progress(i, len(blobs))
        if not blob:
            continue
        try:
            url = _to_data_url(blob)
            if url is None:
                errors.append(f"slide {i}: image too large to send")
                continue
            said = _vision_read(url, state)
            if said and said.strip() and "NO_TEXT" not in said:
                parts.append(f"[Slide {i}]\n{said.strip()}")
        except Exception as e:
            errors.append(f"slide {i}: {type(e).__name__}: {e}")

    return "\n\n".join(parts), errors


def _pdf_text(raw):
    """Per-page extraction. One malformed page in a long PDF should cost that
    page, not the document."""
    from pypdf import PdfReader
    pages = []
    for pg in PdfReader(io.BytesIO(raw)).pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def extract_text(uploaded):
    """Read an uploaded CV / JD / notes / slide deck.

    Returns (text, error, note).
      error — a short human-readable reason the file could not be opened,
              or None.
      note  — when the file opened but yielded no text, what was actually
              found instead (e.g. a deck of images). None otherwise.

    The two outcomes are deliberately distinct. "I could not open this" and
    "I opened it and there is no text in it" need different messages — a
    scanned PDF or an all-images deck is not a broken file, and telling
    someone it is sends them off re-exporting a file that was fine.

    This stays in Python rather than moving to the browser like the Worker
    ports: Streamlit runs a real process, so pypdf / docx2txt / python-pptx
    work as they did on the Space. Each accepts a file-like object, so the
    upload is never written to disk."""
    if uploaded is None:
        return "", None, None

    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        raw = uploaded.getvalue()
    except Exception:
        try:
            raw = uploaded.read()
        except Exception as e:
            return "", f"could not read the upload ({e})", None

    if not raw:
        return "", "the file is empty", None

    try:
        if name.endswith(".pdf"):
            text = _pdf_text(raw)
        elif name.endswith(".docx"):
            import docx2txt
            text = docx2txt.process(io.BytesIO(raw)) or ""
        elif name.endswith(".pptx"):
            text = _pptx_text(raw)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        # Surfaced to the user, not just printed. The first version swallowed
        # this and showed a flat "Could not read that file", which made a
        # real failure impossible to diagnose from the app.
        print("extract error:", type(e).__name__, e)
        return "", f"{type(e).__name__}: {e}", None

    if text.strip():
        return text, None, None

    # Opened cleanly, no text. Work out what IS in there so the message can
    # say something true rather than offering the user a list of guesses.
    note = None
    if name.endswith(".pptx"):
        slides, images = _pptx_inventory(raw)
        if slides or images:
            note = (f"the deck is {slides} slide{'s' if slides != 1 else ''} and "
                    f"{images} image{'s' if images != 1 else ''} with no text at all — "
                    "slides exported as pictures (NotebookLM, Canva and similar tools "
                    "do this) have nothing to read")
    elif name.endswith(".pdf"):
        note = "the PDF has no text layer, which usually means it is a scan"
    return "", None, note


def _chunk(text, size=800):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > size:
            out.append(cur.strip())
            cur = ""
        cur += " " + w
    if cur.strip():
        out.append(cur.strip())
    return out


def build_context(query, full_text, k=4):
    """Small doc -> whole. Large doc -> top-k BM25 chunks for this query."""
    if not full_text or not full_text.strip():
        return ""
    if len(full_text) <= SMALL_DOC:
        return full_text
    chunks = _chunk(full_text)
    if not chunks:
        return ""
    bm = BM25Okapi([c.lower().split() for c in chunks])
    q = (query or "").lower().split() or ["context"]
    return "\n\n".join(bm.get_top_n(q, chunks, n=min(k, len(chunks))))


# ---------------------------------------------------------------- helpers
def transcript_from(msgs):
    return "\n".join(("User: " if m["role"] == "user" else "Coach: ") + m["content"] for m in msgs)


def history_payload(history):
    """Shape of the Download History file. The schema tag lets a future
    version reject or migrate an old export instead of silently misreading it."""
    return {"schema": HISTORY_SCHEMA, "exported": int(time.time()), "conversations": history}


def parse_history_upload(raw):
    """Read a previously downloaded history file. Returns (conversations, error)."""
    try:
        data = json.loads(raw)
    except Exception:
        return None, "That file is not valid JSON."
    if isinstance(data, list):
        return data, None                      # tolerate a bare list
    if not isinstance(data, dict):
        return None, "Unrecognised file format."
    if data.get("schema") not in (None, HISTORY_SCHEMA):
        return None, f"Unrecognised history version: {data.get('schema')}"
    convs = data.get("conversations")
    if not isinstance(convs, list):
        return None, "No conversations found in that file."
    return convs, None


# ---------------------------------------------------------------- session state
# Streamlit reruns the script top to bottom on every interaction, so anything
# that must survive a rerun lives here and nowhere else.
DEFAULTS = {
    "msgs": [],             # live conversation: [{role, content}]
    "history": [],          # ended conversations, newest first
    "doctxt": "",           # extracted document text
    "last_fb": "",          # most recent feedback, for Translate
    "reply_audio": None,    # bytes of the coach's latest reply
    "last_audio_id": "",    # fingerprint of the last processed recording
    "vision_text": "",      # slides transcribed by the vision model, if any
    "section": "Call",
    "notice": None,         # (kind, message) shown once after a rerun
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def run_turn(user_text, cfg):
    """One exchange. Returns True if a reply was produced."""
    user_text = (user_text or "").strip()
    if not user_text:
        return False

    ctx = cfg["ctx_paste"] if cfg["ctx_mode"] == "Paste text" \
        else build_context(user_text, st.session_state.doctxt)

    st.session_state.msgs.append({"role": "user", "content": user_text})
    try:
        reply = chat_logic(st.session_state.msgs, cfg["mode"], cfg["persona"],
                           cfg["target"], cfg["native"], cfg["vocab"], ctx)
    except Exception as e:
        # Roll the user turn back so a failed call doesn't leave a dangling
        # message that the next successful turn would answer out of order.
        st.session_state.msgs.pop()
        st.session_state.notice = ("error", f"Could not reach the model: {e}")
        return False

    st.session_state.msgs.append({"role": "assistant", "content": reply})
    st.session_state.reply_audio = speak(reply, CODE.get(cfg["target"], "en"))
    return True


# ---------------------------------------------------------------- sidebar: setup
st.sidebar.title("🎙️ VoxCoach")
st.sidebar.caption("Speaking practice — languages, interviews, presentations, meetings.")

if get_client() is None:
    st.sidebar.error("GROQ_API_KEY is not set. Add it under Settings → Secrets.")

st.sidebar.markdown("#### What do you want to practise?")
mode = st.sidebar.radio("Mode", ["Language", "Interview", "Presentation", "Meeting"], index=0)

st.sidebar.markdown("#### Set the scene")
persona = st.sidebar.selectbox("Persona", PERSONA_NAMES, index=PERSONA_NAMES.index("Friendly Tutor"))
target = st.sidebar.selectbox("Target language (learning)", LANG_NAMES,
                              index=LANG_NAMES.index("English (US)"))
native = st.sidebar.selectbox("Native language (fluent)", LANG_NAMES,
                              index=LANG_NAMES.index("Urdu"))
vocab = st.sidebar.text_area(
    "Target words (Language mode only)", height=80,
    placeholder="market, price, sell, buy, customer, cheap, money, shop")

fb_lang = st.sidebar.selectbox("Feedback language", LANG_NAMES,
                               index=LANG_NAMES.index("English (US)"))

ctx_mode = st.sidebar.radio("Context source (CV / JD / notes / slides)",
                            ["Paste text", "Upload document"], index=0)
ctx_paste = ""
if ctx_mode == "Paste text":
    ctx_paste = st.sidebar.text_area("Paste context", height=120)
else:
    up = st.sidebar.file_uploader("Upload document", type=["pdf", "docx", "pptx", "txt"])
    if up is not None:
        text, err, note = extract_text(up)
        if err:
            st.sidebar.error(f"Could not read that file — {err}")
        elif text.strip():
            st.session_state.doctxt = text
            st.sidebar.success(f"Loaded document ({len(text):,} characters).")
        elif st.session_state.vision_text:
            # Already transcribed this deck earlier in the session.
            st.session_state.doctxt = st.session_state.vision_text
            st.sidebar.success(
                f"Read from the slides ({len(st.session_state.vision_text):,} characters).")
        else:
            # Opened fine, genuinely contains no extractable text. Saying
            # "could not read" here would send someone off re-exporting a
            # file that was never the problem, so say what is actually in it.
            st.sidebar.warning(
                (f"No text to extract — {note}." if note
                 else "Opened that file, but found no text in it.")
            )

            # A picture deck can still be read — by looking at it. One API
            # call per slide, so it is a button with the cost stated, never
            # something that happens on upload.
            raw_bytes = up.getvalue()
            slides, images = _pptx_inventory(raw_bytes)
            if up.name.lower().endswith(".pptx") and images:
                st.sidebar.caption(
                    f"These slides are pictures. VoxCoach can read them with a "
                    f"vision model — about {slides} API call"
                    f"{'s' if slides != 1 else ''}, roughly "
                    f"{max(1, round(slides * 1.5 / 60))} min."
                )
                if st.sidebar.button("🔍 Read the slides with AI", type="primary"):
                    bar = st.sidebar.progress(0.0, text="Reading slide 1…")

                    def tick(i, total):
                        bar.progress(i / max(total, 1), text=f"Reading slide {i} of {total}…")

                    got, errs = read_slides_with_vision(raw_bytes, progress=tick)
                    bar.empty()
                    if got.strip():
                        st.session_state.vision_text = got
                        st.session_state.doctxt = got
                        st.sidebar.success(f"Read {len(got):,} characters from the slides.")
                    else:
                        st.sidebar.error("Could not read any text from these slides.")
                    for e in errs[:3]:
                        st.sidebar.caption(f"⚠️ {e}")
                    st.rerun()
                else:
                    st.sidebar.caption("Or switch to **Paste text** and paste the key points.")

CFG = {"mode": mode, "persona": persona, "target": target, "native": native,
       "vocab": vocab, "ctx_mode": ctx_mode, "ctx_paste": ctx_paste}

# ---------------------------------------------------------------- main
# NOTE: this radio deliberately has NO `key=`.
# With key="section", Streamlit treats st.session_state["section"] as the
# widget's own state and refuses any assignment to it later in the same run:
#   StreamlitWidgetAlreadyInstantiatedError
# "End & Save conversation" does exactly that — it jumps the user to Feedback
# after the conversation is saved — so the app would have crashed at the one
# moment it matters most, immediately after paying for the feedback call.
# Driving the radio from `index` instead keeps the state key ordinary, so it
# can be reassigned anywhere and picked up on the next rerun.
SECTIONS = ["Call", "History", "Feedback"]
section = st.radio("Section", SECTIONS,
                   index=SECTIONS.index(st.session_state.section),
                   horizontal=True, label_visibility="collapsed")
st.session_state.section = section

if st.session_state.notice:
    kind, msg = st.session_state.notice
    (st.error if kind == "error" else st.info)(msg)
    st.session_state.notice = None

# ---- Call ----
if section == "Call":
    st.markdown("#### 📞 Call")
    st.caption("Tap the mic, speak, then stop. The coach replies out loud. "
               "You can also type instead.")

    audio = st.audio_input("Record your turn")

    # The rerun guard. Same recording on a rerun -> same fingerprint -> skip.
    if audio is not None:
        data = audio.getvalue()
        fid = _audio_fingerprint(data)
        if fid and fid != st.session_state.last_audio_id:
            st.session_state.last_audio_id = fid
            with st.spinner("Listening…"):
                said = transcribe(data, getattr(audio, "name", "speech.wav"),
                                  CODE.get(target, "en"))
            if not said:
                st.session_state.notice = ("error", "Could not make out that recording — try again.")
                st.rerun()
            with st.spinner("Thinking…"):
                run_turn(said, CFG)
            st.rerun()

    with st.form("type_turn", clear_on_submit=True):
        typed = st.text_input("…or type here instead", label_visibility="collapsed",
                              placeholder="…or type here instead")
        sent = st.form_submit_button("Send", type="primary")
    if sent and typed.strip():
        with st.spinner("Thinking…"):
            run_turn(typed, CFG)
        st.rerun()

    # One audio element, always the latest reply. Rendering a player per turn
    # would give several autoplay elements on the page at once, which collide.
    if st.session_state.reply_audio:
        st.audio(st.session_state.reply_audio, format="audio/mp3", autoplay=True)

    for m in st.session_state.msgs:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.write(m["content"])

    c1, c2 = st.columns(2)
    if c1.button("End & Save conversation", type="primary",
                 disabled=not st.session_state.msgs):
        with st.spinner("Writing your feedback…"):
            try:
                fb = feedback_logic(transcript_from(st.session_state.msgs), fb_lang)
            except Exception as e:
                fb = ""
                st.session_state.notice = ("error", f"Could not generate feedback: {e}")
        if fb:
            st.session_state.history.insert(0, {
                "ts": int(time.time()), "mode": mode, "persona": persona,
                "target": target, "native": native,
                "messages": st.session_state.msgs, "feedback": fb,
            })
            st.session_state.last_fb = fb
            st.session_state.msgs = []
            st.session_state.reply_audio = None
            st.session_state.section = "Feedback"
        st.rerun()

    if c2.button("Clear (discard)", disabled=not st.session_state.msgs):
        st.session_state.msgs = []
        st.session_state.reply_audio = None
        st.rerun()

# ---- History ----
elif section == "History":
    st.markdown("#### 🕘 History")
    st.warning(
        "Conversations are kept for this browser session only and are cleared when the "
        "page reloads or the app goes to sleep. Nothing is stored on the server. "
        "**Download** to keep them, and **Restore** to bring them back later.",
        icon="⚠️",
    )

    hist = st.session_state.history
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download history (.json)",
        data=json.dumps(history_payload(hist), ensure_ascii=False, indent=2),
        file_name=f"voxcoach-history-{time.strftime('%Y%m%d')}.json",
        mime="application/json",
        disabled=not hist,
    )
    restore = c2.file_uploader("⬆️ Restore a history file", type=["json"],
                               label_visibility="collapsed")
    if restore is not None:
        convs, err = parse_history_upload(restore.getvalue().decode("utf-8", errors="ignore"))
        if err:
            st.error(err)
        else:
            # Merge rather than replace, and drop exact duplicates by timestamp,
            # so restoring twice doesn't double the list.
            seen = {c.get("ts") for c in st.session_state.history}
            added = [c for c in convs if c.get("ts") not in seen]
            st.session_state.history = sorted(
                st.session_state.history + added,
                key=lambda c: c.get("ts", 0), reverse=True)
            st.success(f"Restored {len(added)} conversation(s).")

    if not st.session_state.history:
        st.info("No saved conversations yet.")
    else:
        for c in st.session_state.history[:30]:
            when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(c.get("ts", 0)))
            title = f"{c.get('mode','')} · {c.get('persona','')} · {c.get('target','')} · {when} UTC"
            with st.expander(title, expanded=False):
                for m in c.get("messages", []):
                    st.markdown(("**You:** " if m["role"] == "user" else "**Coach:** ") + m["content"])
                if c.get("feedback"):
                    st.markdown("**Feedback:**")
                    st.markdown(c["feedback"])

# ---- Feedback ----
else:
    st.markdown("#### 📝 Feedback")
    if not st.session_state.last_fb:
        st.info("End a conversation on **Call** to generate feedback here.")
    else:
        st.markdown(st.session_state.last_fb)
        if st.button(f"Translate feedback → {target}"):
            with st.spinner("Translating…"):
                try:
                    st.session_state.last_fb = translate_logic(st.session_state.last_fb, target)
                except Exception as e:
                    st.session_state.notice = ("error", f"Could not translate: {e}")
            st.rerun()