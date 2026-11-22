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

import io
import json
import os
import time
import hashlib

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
    return "\n".join(out)


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

    Returns (text, error). `error` is a short human-readable reason the file
    could not be opened at all, or None.

    The two outcomes are deliberately distinct. "I could not open this" and
    "I opened it and there is no text in it" need different messages — a
    scanned PDF or an all-images deck is not a broken file, and telling
    someone it is sends them off re-exporting a file that was fine.

    This stays in Python rather than moving to the browser like the Worker
    ports: Streamlit runs a real process, so pypdf / docx2txt / python-pptx
    work as they did on the Space. Each accepts a file-like object, so the
    upload is never written to disk."""
    if uploaded is None:
        return "", None

    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        raw = uploaded.getvalue()
    except Exception:
        try:
            raw = uploaded.read()
        except Exception as e:
            return "", f"could not read the upload ({e})"

    if not raw:
        return "", "the file is empty"

    try:
        if name.endswith(".pdf"):
            return _pdf_text(raw), None
        if name.endswith(".docx"):
            import docx2txt
            return (docx2txt.process(io.BytesIO(raw)) or ""), None
        if name.endswith(".pptx"):
            return _pptx_text(raw), None
        return raw.decode("utf-8", errors="ignore"), None
    except Exception as e:
        # Surfaced to the user, not just printed. The first version swallowed
        # this and showed a flat "Could not read that file", which made a
        # real failure impossible to diagnose from the app.
        print("extract error:", type(e).__name__, e)
        return "", f"{type(e).__name__}: {e}"


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
        text, err = extract_text(up)
        if err:
            st.sidebar.error(f"Could not read that file — {err}")
        elif text.strip():
            st.session_state.doctxt = text
            st.sidebar.success(f"Loaded document ({len(text):,} characters).")
        else:
            # Opened fine, genuinely contains no extractable text. Saying
            # "could not read" here would send someone off re-exporting a
            # file that was never the problem.
            st.sidebar.warning(
                "Opened that file, but found no text in it. If it is a scan or a "
                "deck of image slides there is nothing to extract — paste the key "
                "points instead."
            )

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