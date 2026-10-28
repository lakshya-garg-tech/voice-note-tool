import hashlib
import json
import os

import streamlit as st
from groq import Groq

st.set_page_config(page_title="Voice Notes", page_icon="🎙️", layout="wide")

# ---------- Config (edit here only) ----------
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "openai/gpt-oss-120b"  # keep whichever model worked for you
MAX_AUDIO_MB = 25

TEMPLATES = {
    "General notes": ["summary", "key_points", "decisions", "action_items", "open_questions"],
    "Doctor - consultation": ["complaint", "findings", "diagnosis", "recommended_action", "follow_up"],
    "Agent - site visit": ["property", "client_requirements", "observations", "objections", "next_steps"],
    "Accountant - client call": ["client", "topics_discussed", "issues_raised", "action_items", "deadlines"],
}

LANGS = {
    "English": "en", "Hindi": "hi", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Punjabi": "pa", "Urdu": "ur",
    "Spanish": "es", "French": "fr", "German": "de", "Arabic": "ar",
}
LANGS_IN = {"Auto-detect": None, **LANGS}

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

.stApp { font-family: 'Inter', sans-serif; }
footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3rem; }

.masthead { margin-bottom: 1.4rem; }
.masthead .title { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 600; letter-spacing: -0.01em; }
.masthead .sub { opacity: .65; margin-top: .25rem; }

.label { font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .6; margin-bottom: .4rem; }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; box-shadow: none; }
textarea { font-size: 15.5px !important; line-height: 1.6 !important; }
textarea::placeholder { opacity: .55; }
.stButton > button, .stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 500; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Setup ----------
def get_client():
    key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    if not key:
        st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your host's secrets.")
        st.stop()
    return Groq(api_key=key)


client = get_client()
ss = st.session_state


# ---------- Core functions ----------
def transcribe(audio_bytes: bytes, lang_code):
    kwargs = {"language": lang_code} if lang_code else {}
    result = client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes),
        model=STT_MODEL,
        response_format="json",
        **kwargs,
    )
    return (result.text or "").strip()


def to_text(v) -> str:
    if isinstance(v, list):
        return "\n".join(f"- {to_text(x)}" for x in v)
    if isinstance(v, dict):
        return "\n".join(f"{k}: {to_text(x)}" for k, x in v.items())
    return "" if v is None else str(v).strip()


def structure(transcript: str, fields: list, in_lang: str, out_lang: str) -> str:
    spoken = "in an unknown language (detect it)" if in_lang == "Auto-detect" else f"in {in_lang}"
    system = (
        "You convert a spoken transcript into a structured note. "
        f"The transcript is {spoken}. Write the note in {out_lang}. "
        'Return ONLY a JSON object: {"labels": {...}, "values": {...}}. '
        f"Both objects must contain exactly these keys: {', '.join(fields)}. "
        f'"labels": a short heading for each key, translated into {out_lang}. '
        f'"values": concise text taken from the transcript, written in {out_lang}. '
        'For several points, put each on its own line starting with "- ". '
        "Fill a field ONLY if the speaker clearly talked about that topic. "
        "Never stretch content to fit a field, and never add advice, steps or details the speaker did not say. "
        "If a field does not apply, use an empty string. "
        "If a field does not apply, don;t print it."
        "try keeping it as short and straigth as possible without missing any point"
        "Keep names, numbers, drug names and dates exact."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    labels, values = data.get("labels", {}), data.get("values", {})
    blocks = []
    for f in fields:
        body = to_text(values.get(f))
        if not body:
            continue  # skip sections with nothing in them
        label = to_text(labels.get(f)) or f.replace("_", " ").capitalize()
        blocks.append(f"{label}:\n{body}")
    return "\n\n".join(blocks)


def make_note():
    note = structure(ss["transcript"], TEMPLATES[ss["template"]], ss["lang_in"], ss["lang_out"])
    ss["note_text"] = note
    if not note:
        ss["error"] = (
            "Nothing in the recording fit this use case. "
            "Try 'General notes' or pick a different one, then press Regenerate."
        )


def regenerate():  # button callback
    try:
        ss["error"] = None
        make_note()
    except Exception as e:
        ss["error"] = str(e)


# ---------- UI ----------
st.markdown(
    '<div class="masthead"><div class="title">Voice notes</div>'
    '<div class="sub">Speak after a meeting or visit. Get a clean note you can edit and save.</div></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Use case", list(TEMPLATES.keys()), key="template")
    c2.selectbox("I will speak in", list(LANGS_IN.keys()), key="lang_in")
    c3.selectbox("Write the note in", list(LANGS.keys()), key="lang_out")
    audio = st.audio_input("Press the microphone, speak, then press stop")

# Process each new recording once
if audio is not None:
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if ss.get("last_audio") != digest:
        ss["last_audio"] = digest
        ss["error"] = None
        if len(data) > MAX_AUDIO_MB * 1024 * 1024:
            ss["error"] = "Recording too long. Keep it under a few minutes."
        else:
            try:
                with st.spinner("Transcribing..."):
                    ss["transcript"] = transcribe(data, LANGS_IN[ss["lang_in"]])
                if not ss["transcript"]:
                    ss["error"] = "No speech detected. Try again."
                else:
                    with st.spinner("Writing the note..."):
                        make_note()
            except Exception as e:
                ss["error"] = str(e)

if ss.get("error"):
    st.warning(ss["error"])

st.write("")
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="label">What you said</div>', unsafe_allow_html=True)
        st.text_area(
            "Transcript", key="transcript", height=420, label_visibility="collapsed",
            placeholder="The transcript shows up here after you record. You can fix any wrong word.",
        )

with right:
    with st.container(border=True):
        st.markdown('<div class="label">Note</div>', unsafe_allow_html=True)
        st.text_area(
            "Note", key="note_text", height=420, label_visibility="collapsed",
            placeholder="Your structured note shows up here. Edit anything before saving.",
        )
        has_text = bool(ss.get("transcript"))
        b1, b2 = st.columns(2)
        b1.button("Regenerate", on_click=regenerate, disabled=not has_text, type="primary")
        b2.download_button(
            "Save as text file", data=ss.get("note_text", ""), file_name="note.txt",
            disabled=not ss.get("note_text"),
        )

st.caption("Changed the language you speak in? Record again. Changed the use case or note language? Press Regenerate.")