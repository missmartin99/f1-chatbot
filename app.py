"""Pit Wall Pal: Streamlit front end for the F1 RAG chatbot.

The browser only ever talks to Streamlit. This server calls the n8n webhook, so the
n8n address and any secret stay on the server (st.secrets / environment variables).

Run locally:   .venv/bin/streamlit run app.py
Developer view: add ?debug=1 to the URL to see retrieval details under each answer.
"""
from __future__ import annotations

import html
import os
import re
import time
from datetime import datetime

import requests
import streamlit as st

# ----------------------------------------------------------------- config
MAX_QUESTION_CHARS = 600
MAX_QUESTIONS_PER_SESSION = 40
REQUEST_TIMEOUT_S = 60
FRIENDLY_ERROR = "The pit wall radio is down for a moment. Please try again in a few seconds."

SUGGESTIONS = [
    "What's an undercut?",
    "Why did everyone slow down?",
    "What does parc fermé mean?",
    "Is DRS still a thing?",
    "What do the tyre colours mean?",
    "What is Overtake Mode?",
]
SESSIONS = ["(not set)", "Practice", "Qualifying", "Sprint Qualifying", "Sprint", "Race"]
TRACK_STATUS = ["Normal racing", "Yellow flag", "Double yellow flag", "Safety Car",
                "Virtual Safety Car", "Red flag", "Rain"]
STATUS_COLOUR = {"Normal racing": "#00d26a", "Yellow flag": "#ffd400", "Double yellow flag": "#ffd400",
                 "Safety Car": "#ff9f1c", "Virtual Safety Car": "#ff9f1c", "Red flag": "#e10600", "Rain": "#4aa3ff"}


def secret(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets, then the environment, then a default."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file locally
        pass
    return os.environ.get(name, default)


# Two interchangeable back ends:
#   "python" (default when OPENAI_API_KEY is set): the RAG pipeline in rag.py runs inside this app. Free hosting friendly.
#   "n8n": this app forwards questions to the n8n webhook at CHAT_API_URL instead.
OPENAI_API_KEY = secret("OPENAI_API_KEY", "")
BACKEND = secret("BACKEND", "python" if OPENAI_API_KEY else "n8n").lower()
API_URL = secret("CHAT_API_URL", "http://localhost:5678/webhook/f1-rag-chat")
AUTH_HEADER = secret("CHAT_API_AUTH_HEADER", "")   # e.g. "X-Api-Key"
AUTH_VALUE = secret("CHAT_API_AUTH_VALUE", "")

st.set_page_config(page_title="Pit Wall Pal · F1 viewing companion", page_icon="🏁", layout="wide",
                   initial_sidebar_state="collapsed")
DEBUG = "debug" in st.query_params

def html_block(markup: str) -> None:
    """Render raw HTML. Leading indentation is stripped because Markdown would show 4+ spaces as a code block."""
    flat = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


# ----------------------------------------------------------------- styling
CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap");
:root { --red:#e10600; --line:#1e242b; --line2:#2a323b; --muted:#7d8896; --dim:#566170; --panel:#0d1013; --panel2:#12161a;
        --display:"Barlow Condensed","Arial Narrow",Arial,sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace; }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.stApp { background: linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px) 0 0/100% 4px,
         radial-gradient(90% 60% at 15% -10%, #161b22 0%, transparent 60%), #07080a; }
.block-container { padding: 0 1rem 1rem !important; max-width: 1560px; }
html, body, [class*="st-"], .stMarkdown, p, label { font-family: "Barlow", system-ui, sans-serif; }

/* top bar */
.pw-top { display:flex; align-items:stretch; gap:20px; background:#0a0c0f; border-bottom:1px solid var(--line);
          margin: 0 -1rem; padding: 0 16px; min-height:52px; }
.pw-brand { display:flex; align-items:center; gap:12px; padding-right:20px; border-right:1px solid var(--line); }
.pw-mark { width:30px; height:30px; background:var(--red); color:#fff; display:grid; place-items:center; font:700 16px var(--display);
           clip-path:polygon(0 0,100% 0,100% 70%,70% 100%,0 100%); }
.pw-brand h1 { font:700 22px var(--display); margin:0; padding:0; letter-spacing:.06em; text-transform:uppercase; line-height:1; color:#e8ebef; }
.pw-brand small { display:block; font:400 10px var(--mono); color:var(--dim); letter-spacing:.12em; text-transform:uppercase; margin-top:3px; }
.pw-ro { display:flex; flex-direction:column; justify-content:center; padding:0 18px; border-right:1px solid var(--line); min-width:0; }
.pw-ro span { font:400 9.5px var(--mono); letter-spacing:.14em; text-transform:uppercase; color:var(--dim); }
.pw-ro b { font:600 20px var(--display); letter-spacing:.04em; text-transform:uppercase; white-space:nowrap; color:#e8ebef; }
.pw-ro.lap b { font:700 18px var(--mono); }
.pw-flag { height:3px; margin: 0 -1rem 14px; }
.pw-spacer { flex:1; }
.pw-clock { align-self:center; font:500 13px var(--mono); color:var(--muted); letter-spacing:.06em; }

/* stage + panels */
.pw-stage { position:relative; aspect-ratio:16/9; border:1px solid var(--line); overflow:hidden; display:grid; place-items:center;
            background:radial-gradient(120% 90% at 20% 0%, #151b23 0%, #090b0e 60%), #090b0e;
            clip-path:polygon(0 0,calc(100% - 10px) 0,100% 10px,100% 100%,10px 100%,0 calc(100% - 10px)); }
.pw-stage:before, .pw-stage:after { content:""; position:absolute; width:22px; height:22px; border:2px solid var(--red); z-index:2; }
.pw-stage:before { top:8px; left:8px; border-right:0; border-bottom:0; }
.pw-stage:after { bottom:8px; right:8px; border-left:0; border-top:0; }
.pw-stage svg { position:absolute; inset:0; width:100%; height:100%; opacity:.55; }
.pw-stage .msg { position:relative; text-align:center; padding:0 24px; max-width:520px; }
.pw-stage .msg strong { display:block; font:600 24px var(--display); letter-spacing:.05em; text-transform:uppercase; margin-bottom:6px; color:#e8ebef; }
.pw-stage .msg span { color:var(--muted); font-size:13px; }
.pw-live { position:absolute; top:16px; left:38px; z-index:3; background:var(--red); color:#fff; font:700 10.5px var(--mono); letter-spacing:.14em; padding:3px 9px; }
.pw-badge { position:absolute; top:16px; right:38px; z-index:3; font:400 10.5px var(--mono); letter-spacing:.1em; text-transform:uppercase; padding:3px 10px; background:#000a; border:1px solid; }
.pw-h2 { font:600 17px var(--display); text-transform:uppercase; letter-spacing:.08em; color:#e8ebef; margin:6px 0 0; border-left:3px solid var(--red); padding-left:10px; }
.pw-hint { color:var(--muted); font-size:12.5px; margin:2px 0 8px 13px; }

/* widgets */
[data-testid="stWidgetLabel"] p { font:400 10px var(--mono) !important; letter-spacing:.12em; text-transform:uppercase; color:var(--dim) !important; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background:var(--panel2) !important; border-radius:0 !important; border:1px solid var(--line) !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:var(--red) !important; }
.stButton > button { background:var(--panel2); border:1px solid var(--line2); border-radius:0; color:#e8ebef; font:400 11px var(--mono); padding:.3rem .7rem; }
.stButton > button:hover { border-color:var(--red); color:#fff; background:var(--panel2); }

/* chat */
.pw-chat-title { display:flex; align-items:center; gap:9px; font:700 20px var(--display); letter-spacing:.08em; text-transform:uppercase; color:#e8ebef; }
.pw-chat-title:before { content:""; width:8px; height:8px; background:var(--red); animation:pw 1.4s infinite; }
@keyframes pw { 50% { opacity:.25; } }
[data-testid="stChatMessage"] { background:var(--panel2); border:1px solid var(--line); border-left:3px solid #00d26a; border-radius:0; padding:.7rem .9rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { border-left:1px solid #4a1613; border-right:3px solid var(--red); background:#240a09; }
[data-testid^="stChatMessageAvatar"] { display:none; }
.pw-who { font:400 9.5px var(--mono); letter-spacing:.16em; color:#ff8a84; margin-bottom:2px; }
.pw-cite { display:inline-block; min-width:16px; text-align:center; font:700 10px/16px var(--mono); vertical-align:super; margin:0 1px; background:#1b1330; color:#d3a5f5; border:1px solid #3b2866; padding:0 3px; }
.pw-src { margin-top:9px; font-size:11.5px; color:var(--muted); }
.pw-src i { font:normal 10.5px var(--mono); background:#0b0f14; border:1px solid var(--line2); color:#b9c4d0; padding:2px 8px; margin:0 4px 4px 0; display:inline-block; }
.pw-sys { font:400 11.5px/1.55 var(--mono); color:var(--muted); border-top:1px dashed var(--line2); border-bottom:1px dashed var(--line2); padding:10px 2px; }
[data-testid="stChatInput"] { border-radius:0; }
[data-testid="stChatInput"] textarea { font-family:"Barlow",sans-serif; }
</style>
"""
html_block(CSS)

# ----------------------------------------------------------------- animation layer
def car_svg(body: str, accent: str) -> str:
    """A simple side-on F1 silhouette (generic colours, no real liveries or logos)."""
    return (
        '<svg viewBox="0 0 200 56" aria-hidden="true">'
        f'<rect x="12" y="14" width="3" height="18" fill="#0b0d10"/><rect x="5" y="12" width="17" height="3" fill="{body}"/>'
        f'<rect x="5" y="9" width="2" height="12" fill="{body}"/>'
        f'<path d="M12 32 L30 26 L70 24 L92 14 L108 14 L116 24 L150 28 L184 34 L184 38 L12 38 Z" fill="{body}"/>'
        f'<path d="M58 30 L132 30 L152 33 L58 35 Z" fill="{accent}" opacity=".85"/>'
        '<path d="M92 15 Q104 3 117 17" fill="none" stroke="#0b0d10" stroke-width="2.6" stroke-linecap="round"/>'
        f'<circle cx="103" cy="15" r="4" fill="{accent}"/>'
        f'<rect x="172" y="40" width="26" height="3" fill="{body}"/><rect x="196" y="35" width="2" height="9" fill="{body}"/>'
        '<circle cx="40" cy="34" r="12" fill="#0b0d10" stroke="#2a323b" stroke-width="3"/><circle cx="40" cy="34" r="4" fill="#2a323b"/>'
        '<circle cx="160" cy="36" r="10" fill="#0b0d10" stroke="#2a323b" stroke-width="3"/><circle cx="160" cy="36" r="3.5" fill="#2a323b"/>'
        '</svg>'
    )


# (bottom, width px, seconds per loop, start delay, opacity, blur px, direction, body colour, accent)
CARS = [
    ("1.5vh", 230, 8.0, 1.0, 0.60, 0, "r", "#e10600", "#ffffff"),    # foreground lane, red
    ("1.5vh", 200, 9.5, 12.0, 0.50, 0, "l", "#229971", "#d9f7ea"),   # foreground lane, green, other way
    ("24vh", 150, 15.0, 4.0, 0.16, 1, "l", "#aab4c0", "#e10600"),    # mid lanes: fainter, blurrier, slower
    ("48vh", 115, 20.0, 9.0, 0.11, 2, "r", "#ff8000", "#1b1b1b"),
    ("70vh", 90, 26.0, 14.0, 0.08, 2, "l", "#3671c6", "#ffd400"),
]
BG_HTML = '<div class="pw-bg" aria-hidden="true">' + "".join(
    f'<div class="pw-car {d}" style="bottom:{b};--w:{w}px;--o:{o};--c:{body};animation-duration:{dur}s;'
    f'animation-delay:{delay}s;filter:blur({blur}px)"><div class="pw-streak"></div>{car_svg(body, acc)}</div>'
    for b, w, dur, delay, o, blur, d, body, acc in CARS
) + '<div class="pw-kerb"></div></div>'

# five start lights, each switching on in turn, then all out together
_LIGHT_KEYFRAMES = "".join(
    f"@keyframes pw-l{i}{{0%,{i * 13 + 4}%{{background:#2a1210;box-shadow:none}}"
    f"{i * 13 + 8}%,76%{{background:#ff1a10;box-shadow:0 0 12px 3px rgba(255,26,16,.75)}}"
    f"80%,100%{{background:#2a1210;box-shadow:none}}}}" for i in range(5)
)
_LIGHT_USE = "".join(f".pw-light:nth-child({i + 1}){{animation:pw-l{i} 2.6s linear infinite}}" for i in range(5))

ANIM_CSS = """<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; }
.stApp { background: linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px) 0 0/100% 4px,
         radial-gradient(90% 60% at 15% -10%, #161b22 0%, transparent 60%), #07080a !important; }
.block-container { position: relative; z-index: 1; padding-bottom: 3.2rem !important; }
.pw-bg { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
.pw-car { position: absolute; left: 0; display: flex; align-items: center; opacity: 0; will-change: transform;
          animation-timing-function: linear; animation-iteration-count: infinite; }
.pw-car svg { width: var(--w); height: auto; flex: none; }
.pw-streak { width: calc(var(--w) * .9); height: 2px; margin-right: -8px; background: linear-gradient(90deg, transparent, var(--c)); }
.pw-car.r { animation-name: pw-drive-r; }
.pw-car.l { animation-name: pw-drive-l; flex-direction: row-reverse; }
.pw-car.l svg { transform: scaleX(-1); }
.pw-car.l .pw-streak { margin-right: 0; margin-left: -8px; background: linear-gradient(270deg, transparent, var(--c)); }
@keyframes pw-drive-r { 0% { transform: translateX(-45vw); opacity: 0; } 6% { opacity: var(--o); } 66% { opacity: var(--o); }
                        74% { transform: translateX(115vw); opacity: 0; } 100% { transform: translateX(115vw); opacity: 0; } }
@keyframes pw-drive-l { 0% { transform: translateX(115vw); opacity: 0; } 6% { opacity: var(--o); } 66% { opacity: var(--o); }
                        74% { transform: translateX(-45vw); opacity: 0; } 100% { transform: translateX(-45vw); opacity: 0; } }
.pw-kerb { position: absolute; left: 0; right: 0; bottom: 0; height: 7px; opacity: .55;
           background: repeating-linear-gradient(90deg, #e10600 0 28px, #e8ebef 28px 56px); animation: pw-kerb 1.6s linear infinite; }
@keyframes pw-kerb { to { background-position: 56px 0; } }

/* header: shimmer on the logo, light sweeping along the status strip */
.pw-mark { position: relative; overflow: hidden; }
.pw-mark:after { content: ""; position: absolute; inset: 0; background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.5) 50%, transparent 70%);
                 transform: translateX(-120%); animation: pw-sheen 4.5s ease-in-out infinite; }
@keyframes pw-sheen { 0%, 60% { transform: translateX(-120%); } 100% { transform: translateX(120%); } }
.pw-flag { position: relative; overflow: hidden; }
.pw-flag:after { content: ""; position: absolute; top: 0; bottom: 0; width: 18%; background: linear-gradient(90deg, transparent, rgba(255,255,255,.6), transparent);
                 animation: pw-sweep 3.4s linear infinite; }
@keyframes pw-sweep { from { left: -20%; } to { left: 100%; } }

/* stream box: dashes flow along the circuit and a car-dot laps it */
.pw-trk { animation: pw-dash 1.1s linear infinite; }
@keyframes pw-dash { to { stroke-dashoffset: -12; } }
.pw-dot { offset-path: path("M120 330 C 90 250, 160 170, 260 190 S 330 300, 430 260 S 470 110, 590 120 S 720 230, 660 320 S 520 380, 420 350 S 190 400, 120 330 Z");
          offset-rotate: 0deg; animation: pw-lap 9s linear infinite; filter: drop-shadow(0 0 5px #e10600) drop-shadow(0 0 2px #fff); }
@keyframes pw-lap { from { offset-distance: 0%; } to { offset-distance: 100%; } }
.pw-live i { animation: pw 1.4s infinite; }

/* chat: messages slide in, controls glow */
[data-testid="stChatMessage"] { animation: pw-in .38s cubic-bezier(.2,.8,.2,1); transition: border-color .2s, box-shadow .2s; }
[data-testid="stChatMessage"]:hover { box-shadow: 0 0 0 1px #2a323b, 0 6px 22px rgba(0,0,0,.45); }
@keyframes pw-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.stButton > button { transition: border-color .15s, box-shadow .15s, transform .15s; }
.stButton > button:hover { box-shadow: 0 0 14px rgba(225,6,0,.45); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(1px); }
[data-testid="stChatInput"] { transition: box-shadow .2s; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 1px #e10600, 0 0 18px rgba(225,6,0,.35); }

/* "thinking" = F1 start lights */
.pw-lights { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.pw-light { width: 15px; height: 15px; border-radius: 50%; background: #2a1210; border: 1px solid #3a1c1a; }
.pw-lights span { margin-left: 8px; font: 400 11px var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
""" + _LIGHT_KEYFRAMES + _LIGHT_USE + """
@media (prefers-reduced-motion: reduce) {
  .pw-bg { display: none; }
  *, *:before, *:after { animation: none !important; transition: none !important; }
  .pw-light:nth-child(-n+3) { background: #ff1a10; }
}
</style>"""
html_block(ANIM_CSS)
html_block(BG_HTML)

LIGHTS_HTML = '<div class="pw-lights">' + '<div class="pw-light"></div>' * 5 + '<span>Lights out…</span></div>'

# ----------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("messages", [])      # [{"role": "user"|"assistant", "content": str, "sources": [...], "raw": dict}]
ss.setdefault("asked", 0)
ss.setdefault("pending", None)

# ----------------------------------------------------------------- helpers
def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def rich(text: str) -> str:
    """Escape model output, then allow only **bold** and [n] citation markers."""
    t = esc(text)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[(\d+)\]", r'<span class="pw-cite">\1</span>', t)
    return t.replace("\n", "<br>")


@st.cache_resource(show_spinner=False)
def python_backend(api_key: str):
    """Build the OpenAI client and the knowledge-base index once per server process."""
    from openai import OpenAI
    import rag
    client = OpenAI(api_key=api_key, timeout=30, max_retries=2)
    return rag, client, rag.load_knowledge_base(client)


def ask_backend(question: str, context: dict) -> dict:
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-6:]]
    if BACKEND == "python":
        rag, client, kb = python_backend(OPENAI_API_KEY)
        return rag.answer_question(kb, client, question, context, history)
    headers = {"Content-Type": "application/json"}
    if AUTH_HEADER and AUTH_VALUE:
        headers[AUTH_HEADER] = AUTH_VALUE
    r = requests.post(API_URL, headers=headers, timeout=REQUEST_TIMEOUT_S,
                      json={"question": question, "context": context, "history": history})
    r.raise_for_status()
    data = r.json()
    return data[0] if isinstance(data, list) and data else data


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(f'<div class="pw-who">YOU</div>{esc(m["content"])}', unsafe_allow_html=True)
            return
        st.markdown(rich(m["content"]), unsafe_allow_html=True)
        cited = [s for s in m.get("sources", []) if s.get("cited")]
        if cited:
            chips = "".join(f"<i>[{esc(s.get('n'))}] {esc(s.get('title'))}</i>" for s in cited)
            st.markdown(f'<div class="pw-src">Sources: {chips}</div>', unsafe_allow_html=True)
        if DEBUG and m.get("raw"):
            with st.expander("Debug: retrieval"):
                st.write("Searched for:", m["raw"].get("subqueries"))
                st.dataframe(m["raw"].get("sources", []), hide_index=True)


# ----------------------------------------------------------------- top bar (reads the context widgets below)
ctx_event = ss.get("c_event", "")
ctx_session = ss.get("c_session", "Race")
ctx_lap = ss.get("c_lap", "")
ctx_status = ss.get("c_status", "Normal racing")
colour = STATUS_COLOUR.get(ctx_status, "#00d26a")
flashing = "animation:pw 1.1s steps(2) infinite;" if ctx_status != "Normal racing" else ""
session_txt = "—" if ctx_session == "(not set)" else ctx_session

html_block(f"""
<div class="pw-top">
  <div class="pw-brand"><div class="pw-mark">PW</div><div><h1>Pit Wall Pal</h1><small>Race companion</small></div></div>
  <div class="pw-ro"><span>Event</span><b>{esc(ctx_event) or '—'}</b></div>
  <div class="pw-ro"><span>Session</span><b>{esc(session_txt)}</b></div>
  <div class="pw-ro lap"><span>Lap</span><b>{esc(ctx_lap) or '--'}</b></div>
  <div class="pw-ro" style="border-right:0"><span>Track</span><b style="color:{colour}">{esc('Green' if ctx_status == 'Normal racing' else ctx_status)}</b></div>
  <div class="pw-spacer"></div>
  <div class="pw-clock">{datetime.now().strftime('%H:%M')}</div>
</div>
<div class="pw-flag" style="background:{colour};{flashing}"></div>
""")

# ----------------------------------------------------------------- layout
left, right = st.columns([1.7, 1], gap="medium")

with left:
    html_block(f"""
    <div class="pw-stage">
      <svg viewBox="0 0 800 450" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <path d="M120 330 C 90 250, 160 170, 260 190 S 330 300, 430 260 S 470 110, 590 120 S 720 230, 660 320 S 520 380, 420 350 S 190 400, 120 330 Z" fill="none" stroke="#222b36" stroke-width="14" stroke-linecap="round"/>
        <path d="M120 330 C 90 250, 160 170, 260 190 S 330 300, 430 260 S 470 110, 590 120 S 720 230, 660 320 S 520 380, 420 350 S 190 400, 120 330 Z" class="pw-trk" fill="none" stroke="#e10600" stroke-width="2" stroke-dasharray="3 9" stroke-linecap="round"/>
        <circle class="pw-dot" r="5" fill="#ffffff"/>
      </svg>
      <div class="pw-live">LIVE</div>
      <div class="pw-badge" style="color:{colour};border-color:{colour}">{esc('Racing' if ctx_status == 'Normal racing' else ctx_status)}</div>
      <div class="msg"><strong>Your F1 TV / FanCode player goes here</strong>
      <span>This page is the companion sidebar. Set what is on your screen below, and the chatbot will answer with that in mind.</span></div>
    </div>
    <div class="pw-h2">What's on your screen?</div>
    <div class="pw-hint">Optional. Sent with every question so answers fit the moment. Nothing here is read from the stream.</div>
    """)

    c1, c2, c3 = st.columns([2, 1.2, 0.8])
    c1.text_input("Event", key="c_event", placeholder="e.g. Abu Dhabi Grand Prix", max_chars=80)
    c2.selectbox("Session", SESSIONS, index=5, key="c_session")
    c3.text_input("Lap", key="c_lap", placeholder="e.g. 23", max_chars=3)
    c4, c5 = st.columns([1, 2])
    c4.selectbox("Track status", TRACK_STATUS, key="c_status")
    c5.text_input("Something you noticed", key="c_note", placeholder="e.g. everyone is slowing down in a line", max_chars=200)

with right:
    head_l, head_r = st.columns([3, 1])
    head_l.markdown('<div class="pw-chat-title">Pit wall radio</div>', unsafe_allow_html=True)
    if head_r.button("Clear", key="clear"):
        ss.messages = []
        st.rerun()
    st.caption("Ask anything about the race you are watching.")

    box = st.container(height=470, border=False)
    with box:
        if not ss.messages:
            st.markdown('<div class="pw-sys">PIT WALL ONLINE. I explain F1 rules, flags and jargon in plain words. '
                        "No live timing here, so I can't say who is leading, but I can explain what is happening "
                        "on your screen. Tap a question or type your own.</div>", unsafe_allow_html=True)
        for m in ss.messages:
            render_message(m)

    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sug{i}", use_container_width=True):
            ss.pending = s

    typed = st.chat_input("Ask about a rule, flag or term…", max_chars=MAX_QUESTION_CHARS)
    question = (typed or ss.pending or "").strip()
    ss.pending = None

    if question:
        if ss.asked >= MAX_QUESTIONS_PER_SESSION:
            ss.messages.append({"role": "user", "content": question})
            ss.messages.append({"role": "assistant", "content": "You've asked a lot of questions this session. Please come back a little later."})
            st.rerun()
        context = {"event": ss.get("c_event", ""), "session": "" if ss.get("c_session") == "(not set)" else ss.get("c_session", ""),
                   "lap": ss.get("c_lap", ""), "status": "" if ss.get("c_status") == "Normal racing" else ss.get("c_status", ""),
                   "note": ss.get("c_note", "")}
        context = {k: str(v).strip() for k, v in context.items() if str(v).strip()}
        with box:
            render_message({"role": "user", "content": question})
            with st.chat_message("assistant"):
                thinking = st.empty()
                thinking.markdown(LIGHTS_HTML, unsafe_allow_html=True)
                if True:
                    try:
                        t0 = time.time()
                        data = ask_backend(question, context)
                        answer = data.get("answer") or "Sorry, I could not come up with an answer. Try asking another way."
                        reply = {"role": "assistant", "content": answer, "sources": data.get("sources", []), "raw": data}
                    except Exception as e:  # never show backend details to the user
                        if DEBUG:
                            st.exception(e)
                        reply = {"role": "assistant", "content": FRIENDLY_ERROR, "sources": []}
                thinking.empty()
        ss.messages.append({"role": "user", "content": question})
        ss.messages.append(reply)
        ss.asked += 1
        st.rerun()
