# VoxCoach — Full Build Spec (PRD + Roadmap + TRD)
**Pak-Angels Cohort 11 · Mid-Course Hackathon · 6-hour build**
**One product: a Gradio voice bot you talk to for real-time speaking practice.**

*(Current architecture: a single Gradio app on Hugging Face Spaces. This supersedes all earlier
FastAPI + Netlify + Docker drafts.)*

---

## 0. HOW TO USE THIS DOCUMENT
This is the **single source of truth**. If you run out of credits on one AI, paste this whole
file into the next one and say *"This is the full spec, continue from Section X."* It has the
product, the UI, the code, and the build order — nobody re-explains the idea.

Written so a person who can **read** code but not write it from scratch can still build it: every
important function is named, placed, and described, with the tricky parts written out in full.

**Golden rule:** don't invent new UI, file names, or data shapes. If it's not here, ask first.

---

## 0.1 ARCHITECTURE AT A GLANCE (and where the voice lives)
VoxCoach is **one Gradio app** running on a Hugging Face Space. Everything is Python.

```
[ Hugging Face Space : one Gradio app (app.py) ]
   UI            : Gradio Blocks (5 sections), desktop-first + responsive
   Voice IN      : gr.Audio(microphone) -> Groq Whisper  (speech -> text)
   Brain         : Groq openai/gpt-oss-20b  (text -> text)
   Voice OUT     : gTTS  (text -> mp3, auto-played)
   Context       : paste text  OR  upload file -> extract -> lightweight RAG (BM25)
   Private history: gr.BrowserState  (stored in the user's browser localStorage)
        |
        v
   [ Groq API ]  chat + Whisper transcription
```
The model runs on Groq, so the Space is a light proxy — **no GPU needed**. Voice is handled
inside the same app (record → transcribe → reply → speak); there is no separate frontend.

---

# 1. PRD — Product Requirements

## 1.1 Pitch
A private, in-browser **voice partner** to practise *speaking* — a language, a job interview, a
presentation, or a meeting — with written feedback afterwards.

## 1.2 Who it's for
1. **Language learners** (primary) — conversation practice, especially restricted to words they're learning.
2. **Job seekers** — mock interviews grounded in their real CV + job description.
3. **Presenters / professionals** — rehearse a talk or meeting out loud.

## 1.3 Flagship feature
**Vocabulary-restricted conversation.** The user supplies target words; the bot speaks using
only those words + basic glue words, explaining anything new in the user's native language. A
graded tutor, not an open chatbot. **Lead the demo with this.**

## 1.4 The three language roles (distinct)
- **Target language** = the language being **learned**. Conversation + STT + TTS happen in it.
- **Native language** = a language the user speaks **fluently**. Used only for in-chat word hints.
- **Feedback language** = language the sandwich feedback is written in (default: the user's
  choice / browser language). A **"Translate feedback → target language"** button re-renders it
  in the learning language for immersion.

## 1.5 Context input — user's choice of TWO paths (both supported)
For CV / JD / notes / slides / word lists, the user picks whichever suits them:
- **Paste text** — type or paste directly into a text field. Simple, instant.
- **Upload a document** — PDF / DOCX / PPTX / TXT → text is extracted server-side. If the document
  is small it's used whole; if large, a **lightweight RAG** step (BM25 retrieval over chunks)
  pulls the most relevant pieces per turn. No heavy embeddings — matches the free tier.

Both paths feed the same `doc_context` slot in the engine, so the conversation logic is identical
either way.

## 1.6 Feature list — MVP vs Stretch
| # | Feature | MVP / Stretch |
|---|---------|---------------|
| F1 | Modes: Language / Interview / Presentation / Meeting | **MVP** |
| F2 | Persona selection | **MVP** |
| F3 | Target + native language selection | **MVP** |
| F4 | Context: **paste text** | **MVP** |
| F5 | Context: **upload document** (PDF/DOCX/PPTX/TXT) → extract | **MVP** |
| F6 | **Lightweight RAG** (BM25) when the document is large | **MVP** |
| F7 | **Voice conversation** (mic → Whisper → reply → gTTS) | **MVP** |
| F8 | **Vocabulary restriction** | **MVP** |
| F9 | Typed-input fallback | **MVP** (safety net) |
| F10 | **Private history** persisted via `gr.BrowserState` | **MVP** |
| F11 | **Sandwich feedback** after each conversation | **MVP** |
| F12 | **Translate feedback → target language** | **MVP** |
| S1 | Embedding-based RAG upgrade (sentence-transformers) | Stretch |
| S2 | Export a conversation as .txt | Stretch |
| S3 | Level selector (Beginner/Intermediate/Advanced) | Stretch |

## 1.7 Modes
- **Language** — casual chat; applies vocab restriction; native-language word hints.
- **Interview** — bot interviews the user one question at a time (uses CV + JD context).
- **Presentation** — bot is an audience/coach; user presents, bot asks audience questions.
- **Meeting** — bot is a colleague; discusses, pushes back politely, stays professional.

## 1.8 Personas
`Default` · `Friendly Tutor` · `Strict Interviewer` · `Entrepreneur` · `Psychologist` ·
`Casual Native` · `Skeptical Investor`.

## 1.9 Privacy model (say it in the pitch)
- The server is **stateless** — message in, reply out, nothing kept.
- Conversation history + feedback live **only in the user's browser** (`gr.BrowserState`).
- Uploaded documents are parsed **in memory and discarded**; only extracted text (and only while
  the tab is open / in browser storage) is retained client-side.

## 1.10 Assumptions
- Primarily used on **laptop / desktop (big screens)**; the UI must still **reflow to smaller screens**.
- Demoed on desktop Chrome. No login, no server database.

---

# 2. UI SPECIFICATION (Gradio, desktop-first + responsive)

## 2.1 Layout
- Built with **Gradio Blocks**. **Desktop-first**: a wide two-column working area, which Gradio
  **reflows to a single stacked column on narrow screens** automatically (Rows collapse). Add a
  small CSS media query only if needed.
- Suggested desktop layout:
  - **Left column** = Section 1 (mode) + Section 2 (setup: persona, languages, context).
  - **Right column** = Section 3 (Talk).
  - **Full width below** = Section 4 (History) + Section 5 (Feedback).
- Constrain overall width (e.g. `max-width: 1100px; margin: auto`) so it looks intentional on big screens.

## 2.2 Look
- Use a dark Gradio theme (e.g. `gr.themes.Soft()` or `Base()` tuned dark) + a small custom `css`
  string for max-width, section cards, and the mic button. Palette to echo the brand:
  bg `#0F1117`, card `#171A23`, accent `#6C8BFF`, success `#38D9A9`, warn `#FFB454`.
- Section headers as `gr.Markdown("### 1 · What do you want to practise?")` inside `gr.Group`.

## 2.3 Sections (Gradio components)
**Section 1 — Practise what?**
- `gr.Radio(["Language","Interview","Presentation","Meeting"], value="Language")` → `mode`.

**Section 2 — Set the scene**
- `gr.Dropdown(personas, value="Friendly Tutor")` → persona.
- `gr.Dropdown(lang_names, ...)` × 2 → target language, native language.
- `gr.Textbox(lines=2, label="Target words (Language mode)")` → vocabulary (paste), plus optional
  `gr.File` for a `.txt` word list.
- **Context input with a source toggle:**
  `gr.Radio(["Paste text","Upload document"], value="Paste text")` →
  shows either `gr.Textbox(lines=4)` (paste) or `gr.File(file_types=[".pdf",".docx",".pptx",".txt"])`.

**Section 3 — Talk**
- `gr.Audio(sources=["microphone"], type="filepath", label="Hold to speak")` → the mic.
- A conversation view: `gr.Markdown` (or `gr.Chatbot(type="messages")` if the installed Gradio
  supports it — otherwise Markdown, which is version-proof).
- `gr.Audio(autoplay=True, label="Coach")` → plays the spoken reply.
- **Typed fallback:** `gr.Textbox` + `gr.Button("Send")` (F9), always visible.
- `gr.Button("End & Save conversation")`.

**Section 4 — Your history**
- Loaded from `gr.BrowserState` on app load, newest first, each expandable to the full transcript.

**Section 5 — Feedback**
- Shows sandwich feedback for the last saved conversation (in the feedback language).
- `gr.Button("Translate → target language")` re-renders it in the target language.

## 2.4 States
Show status inline (a `gr.Markdown` status line): `Idle → Listening/Transcribing → Thinking →
Speaking → Idle`. Never a blank screen on error — catch and show a red message.

---

# 3. TRD — Technical Requirements

## 3.1 Stack (committed)
- **App + UI:** Gradio (Blocks) on a **Hugging Face Gradio Space**.
- **Brain:** Groq **`openai/gpt-oss-20b`** (llama-3.3-70b/llama-3.1-8b are decommissioned).
- **STT:** Groq Whisper — **`whisper-large-v3-turbo`** (fast, multilingual; confirm current name in Groq docs).
- **TTS:** **gTTS** (free, wide language coverage).
- **Context/RAG:** `pypdf` / `docx2txt` / `python-pptx` for extraction; **`rank-bm25`** for
  lightweight retrieval on large docs (no embeddings, no torch).
- **Private storage:** `gr.BrowserState` (localStorage).
- **Deploy:** Gradio SDK Space. Put it on the **same free tier your working Spaces use** (your
  StudyBuddy runs on ZeroGPU); the app never calls a GPU, it just proxies Groq.
- **Testing:** iterate in Colab first (free), then push to the Space.

## 3.2 gpt-oss REASONING GOTCHA (still applies)
`openai/gpt-oss-20b` reasons before it answers; a small budget gets eaten and returns empty
content. Every chat/feedback/translate call uses `reasoning_effort="low"` +
`max_completion_tokens=1024` (NOT the deprecated `max_tokens`), plus `frequency_penalty` /
`presence_penalty` on the chat turn to stop repetition.

## 3.3 File layout
```
app.py              <- the entire Gradio app (engine + voice + RAG + UI)
requirements.txt
README.md           <- HF frontmatter: sdk: gradio
```

## 3.4 requirements.txt
```
gradio
groq
gTTS
pypdf
docx2txt
python-pptx
rank-bm25
```

## 3.5 README.md frontmatter
```
---
title: VoxCoach
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---
```

## 3.6 Languages table (single source of truth)
`(display name, tts/stt code)`. Display name goes to the LLM; the short code goes to gTTS and
Whisper. Reuse everywhere.
```python
LANGUAGES = [
    ("English (US)","en"), ("English (UK)","en"), ("Urdu","ur"), ("Hindi","hi"),
    ("Arabic","ar"), ("Spanish (Spain)","es"), ("Spanish (Latin Am.)","es"), ("French","fr"),
    ("German","de"), ("Italian","it"), ("Portuguese (Brazil)","pt"), ("Chinese (Mandarin)","zh-CN"),
    ("Japanese","ja"), ("Korean","ko"), ("Russian","ru"), ("Turkish","tr"), ("Bengali","bn"),
]
LANG_NAMES = [n for n, _ in LANGUAGES]
CODE = dict(LANGUAGES)                      # name -> short code
# Whisper wants the bare ISO code; gTTS is happy with these. For Whisper, use code.split("-")[0].
```

## 3.7 THE ENGINE (identical to the tested Colab core)
```python
import os, time
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

PERSONAS = {
    "Default": "",
    "Friendly Tutor": "You are warm, encouraging and patient, like a kind teacher.",
    "Strict Interviewer": "You are a no-nonsense professional interviewer who probes for detail.",
    "Entrepreneur": "You are a busy startup founder who talks about business and growth.",
    "Psychologist": "You are a calm, thoughtful psychologist who asks reflective questions.",
    "Casual Native": "You are a relaxed native speaker chatting like a friend.",
    "Skeptical Investor": "You are a sharp investor who challenges every claim.",
}
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
    if PERSONAS.get(persona): p.append(PERSONAS[persona])
    if mode == "Language":
        p.append(f"When you use a new or difficult word, add a very short explanation in "
                 f"{native_lang} in brackets right after it.")
    if vocabulary.strip():
        p.append("STRICT VOCABULARY CONSTRAINT: Speak using ONLY these target words plus the "
                 "simplest glue words (a, the, is, are, have, do, go, you, I, and, to). Do NOT use "
                 "advanced vocabulary. If you must introduce a new word, explain it in one very "
                 f"simple phrase in {native_lang}. Keep sentences very short. Keep finding NEW "
                 "things to say -- never loop on the same greeting or question.\n"
                 f"TARGET WORDS: {vocabulary.strip()}")
    if doc_context.strip():
        p.append("Use this user-provided context to make the session realistic (e.g. base "
                 "interview questions on the CV and job description):\n\n" + doc_context.strip()[:6000])
    return "\n\n".join(p)

def clean_messages(msgs): return [{"role": m["role"], "content": m["content"]} for m in msgs]

def _complete(messages, temperature=0.7, max_tokens=1024, freq=0.0, pres=0.0):
    r = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature,
        max_completion_tokens=max_tokens, reasoning_effort="low",
        frequency_penalty=freq, presence_penalty=pres)
    return r.choices[0].message.content

def chat_logic(messages, mode, persona, target_lang, native_lang, vocabulary, doc_context):
    sys = build_system_prompt(mode, persona, target_lang, native_lang, vocabulary, doc_context)
    return _complete([{"role":"system","content":sys}] + clean_messages(messages),
                     temperature=0.8, freq=0.6, pres=0.5)

def feedback_logic(transcript, feedback_lang):
    prompt = ("You are a supportive speaking coach. Give SANDWICH feedback on the transcript:\n"
        "1) 2 specific strengths. 2) 2-3 concrete improvements (grammar/vocab/fluency/clarity) "
        "each with a corrected example. 3) One encouraging line + one thing to try next.\n"
        f"Under 200 words, warm and plain. Write it in {feedback_lang}.\n\nTRANSCRIPT:\n" + transcript)
    return _complete([{"role":"user","content":prompt}], temperature=0.5)

def translate_logic(text, target_lang):
    prompt = (f"Translate the following into {target_lang}. Keep meaning and friendly tone. "
              "Output ONLY the translation.\n\n" + text)
    return _complete([{"role":"user","content":prompt}], temperature=0.2)
```

## 3.8 VOICE — Whisper (STT) + gTTS (TTS)
```python
from gtts import gTTS
import tempfile

def transcribe(audio_path, stt_code):
    """audio_path comes from gr.Audio(type='filepath'). Returns the recognised text."""
    if not audio_path:
        return ""
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3-turbo",
            language=stt_code.split("-")[0],   # e.g. "ja" — improves accuracy
        )
    return tr.text.strip()

def speak(text, tts_code):
    """Returns a path to an mp3 for gr.Audio(autoplay=True). gTTS is free, no key."""
    if not text.strip():
        return None
    path = tempfile.mktemp(suffix=".mp3")
    try:
        gTTS(text=text, lang=tts_code).save(path)   # tts_code like "en","ja","es","zh-CN"
        return path
    except Exception:
        return None      # unsupported language -> stay silent, text still shows
```

## 3.9 CONTEXT — extraction + lightweight RAG
```python
from rank_bm25 import BM25Okapi

def extract_text(file_path):
    n = file_path.lower()
    if n.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(file_path).pages)
    if n.endswith(".docx"):
        import docx2txt; return docx2txt.process(file_path) or ""
    if n.endswith(".pptx"):
        from pptx import Presentation
        return "\n".join(sh.text for s in Presentation(file_path).slides
                         for sh in s.shapes if sh.has_text_frame)
    return open(file_path, encoding="utf-8", errors="ignore").read()

SMALL_DOC = 6000   # chars: below this, use the whole document; above, retrieve

def chunk(text, size=800):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > size:
            out.append(cur); cur = ""
        cur += (" " + w)
    if cur.strip(): out.append(cur.strip())
    return out

def build_context(query, full_text, k=4):
    """Small doc -> whole. Large doc -> top-k BM25 chunks for this query (lightweight RAG)."""
    if not full_text.strip(): return ""
    if len(full_text) <= SMALL_DOC: return full_text
    chunks = chunk(full_text)
    bm = BM25Okapi([c.lower().split() for c in chunks])
    ranked = bm.get_top_n(query.lower().split(), chunks, n=k)
    return "\n\n".join(ranked)
```
- **Paste path:** the pasted text IS the context (used whole, truncated to 6000 chars in the prompt).
- **Upload path:** `extract_text(file)` once, keep the full text in `gr.State`; each turn call
  `build_context(latest_user_message, full_text)` to get the context for that turn.

## 3.10 GRADIO APP structure (Blocks + BrowserState)
```
STATE
  msgs   = gr.State([])                         # current conversation [{role,content}]
  doctxt = gr.State("")                         # extracted full document text (upload path)
  history = gr.BrowserState([], storage_key="voxcoach_history")   # saved conversations (localStorage)
  last_feedback = gr.State("")

HANDLERS
  on_upload(file)         -> doctxt = extract_text(file.name)   ; status "Document loaded"
  on_voice(audio, ...)    -> text = transcribe(audio, stt_code) ; on_user_text(text, ...)
  on_user_text(text, msgs, mode, persona, tgtName, natName, vocab, ctx_mode, ctx_paste, doctxt, tts_code):
        ctx = ctx_paste if ctx_mode=="Paste text" else build_context(text, doctxt)
        msgs = msgs + [{"role":"user","content":text}]
        reply = chat_logic(msgs, mode, persona, tgtName, natName, vocab, ctx)
        msgs = msgs + [{"role":"assistant","content":reply}]
        return render(msgs), speak(reply, tts_code), msgs        # -> Markdown, Audio(autoplay), State
  on_end(msgs, history, ...)  -> transcript -> feedback_logic(..) ; append {ts,mode,persona,messages,feedback}
        to history (BrowserState) ; clear msgs ; show feedback ; refresh history view
  on_translate(last_feedback, tgtName) -> translate_logic(last_feedback, tgtName)

LOAD
  demo.load(render_history, inputs=history, outputs=history_view)   # BrowserState hydrates from localStorage
```
- Store timestamps as `int(time.time())`; format for display with `time.strftime` (note: the Space
  runs in UTC — fine for a demo; if exact local time matters, format client-side).
- `gr.BrowserState` keeps history in the browser, so it survives reloads and never hits the server —
  this is what preserves the privacy pitch.

## 3.11 Config & gotchas
- `GROQ_API_KEY` → Space **Secrets** (never in code).
- README `sdk: gradio` + `app_file: app.py`.
- Model `openai/gpt-oss-20b`; Whisper `whisper-large-v3-turbo`; reasoning fix from 3.2.
- gTTS language codes are the short codes from the LANGUAGES table (`en, ur, ja, zh-CN, ...`).
- Deploy as a Gradio Space on a free tier that works on your account (ZeroGPU tier is fine; the
  app never uses the GPU). Docker/cpu-basic is the quota-blocked path — avoid it.
- Recording needs mic permission (browser prompts). Demo on desktop Chrome.

---

# 4. ROADMAP (6-hour hackathon) — with status
*(This is the shareable roadmap; a standalone copy also exists for the mentor.)*

| Phase | Work | Status |
|-------|------|--------|
| 1 | **Conversation engine** — prompt builder, modes, personas, vocab restriction, feedback, translate; tested in Colab | ✅ Done & tested |
| 2 | **Voice I/O** — Groq Whisper (STT) + gTTS (TTS) | 🔨 In progress |
| 3 | **Context + lightweight RAG** — paste path + upload→extract→BM25 | 🔨 In progress |
| 4 | **Gradio UI** — 5 sections, desktop-first + responsive | 🔨 In progress |
| 5 | **Private history + feedback + translate** — wire `gr.BrowserState` | ⬜ Next |
| 6 | **Deploy to HF (Gradio Space)** + `GROQ_API_KEY` secret | ⬜ Next |
| 7 | **Polish + demo** — states, errors, rehearse, record backup | ⬜ Next |

**Cut-lines (drop in order if behind):** embedding RAG upgrade → export → persona → translate →
feedback. **Never drop:** the voice loop + vocabulary restriction — that's the product.

---

# 5. DEMO SCRIPT (2 minutes)
1. Pick **Language**; set target = learning language, native = fluent one; paste ~20 target words.
2. Speak a sentence → bot replies **out loud, only using those words**, glossing new ones. ← wow moment.
3. Switch to **Interview**; **upload** a job description (shows the RAG path); do one spoken Q&A round.
4. End the session → **History** shows the full transcript → **Feedback** appears → tap **Translate**.
5. Refresh the page → history persists → *"nothing left our server."*

# 6. TEST / DEMO CHECKLIST
- [ ] Space boots (Gradio UI loads).
- [ ] Language + word list → bot stays inside the words, glosses new ones, doesn't loop.
- [ ] Mic → Whisper transcribes correctly in the target language.
- [ ] Reply is spoken via gTTS in the target language.
- [ ] Paste context works; upload → extract works; large upload → RAG retrieves relevant parts.
- [ ] Interview + uploaded JD → relevant questions.
- [ ] End → transcript in History → feedback → translate → survives page refresh.
- [ ] `GROQ_API_KEY` secret set; model + Whisper names correct; reasoning fix in place.
- [ ] Backup screen recording saved.

# 7. PRIVACY
- Stateless backend; nothing user-facing stored on the server.
- History + feedback stored in the user's browser via `gr.BrowserState`.
- Uploaded documents parsed in memory and discarded; only extracted text is retained client-side.

---
*End of spec. Everything needed to build and hand off VoxCoach is above.*
