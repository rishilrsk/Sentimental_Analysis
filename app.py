import html
import re

import streamlit as st

from sentiment import clean_text, key_words, load_model, predict

st.set_page_config(page_title="How does this sentence sound?", layout="centered")

ORDER = ("positive", "neutral", "negative")
EXAMPLES = (
    "I really enjoyed the movie last night!",
    "The meeting is scheduled for 3 PM tomorrow.",
    "I hate waiting in long lines.",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,500;7..72,600&family=Public+Sans:wght@400;500;600&display=swap');

:root {
  --paper: #F2F4F6;
  --surface: #FFFFFF;
  --ink: #18212E;
  --muted: #5E6A7A;
  --rule: #D6DCE3;
  --positive: #1E7B55;
  --neutral: #4F6D8F;
  --negative: #B23A48;
  --serif: 'Literata', Georgia, serif;
  --sans: 'Public Sans', system-ui, sans-serif;
}

/* Hide Streamlit's own chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

.stApp { background: var(--paper); }
.stApp, .stApp p, .stApp button, .stApp textarea { font-family: var(--sans); color: var(--ink); }
.block-container { max-width: 44rem; padding: 5rem 1.5rem 4rem; }

h1.title {
  font-family: var(--serif); font-weight: 500; font-size: 2.6rem; line-height: 1.15;
  letter-spacing: -0.01em; margin: 0 0 0.75rem; padding: 0; color: var(--ink);
}
p.lede { font-size: 1.06rem; line-height: 1.6; color: var(--muted); max-width: 34rem; margin: 0 0 2.25rem; }

/* Text box */
.stTextArea [data-baseweb="textarea"] {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
}
.stTextArea [data-baseweb="textarea"]:focus-within { border-color: var(--ink); }
.stTextArea textarea {
  font-family: var(--serif) !important; font-size: 1.25rem; line-height: 1.55;
  color: var(--ink); background: var(--surface); padding: 0.9rem 1rem;
}
.stTextArea textarea::placeholder { color: #9AA4B1; }

/* Primary button */
.stButton button[kind="primary"] {
  background: var(--ink); border: 1px solid var(--ink); color: #fff; border-radius: 6px;
  font-weight: 600; padding: 0.55rem 1.25rem;
}
.stButton button[kind="primary"]:hover { background: #2B3748; border-color: #2B3748; }
.stButton button[kind="primary"] p { color: #fff; }

/* Example sentences as quiet text buttons */
.stButton button[kind="tertiary"] {
  color: var(--muted); padding: 0; min-height: 0; text-align: left;
  text-decoration: underline; text-decoration-color: var(--rule); text-underline-offset: 4px;
}
.stButton button[kind="tertiary"] p { color: var(--muted); font-size: 0.95rem; }
.stButton button[kind="tertiary"]:hover p { color: var(--ink); }
.stApp button:focus-visible { outline: 2px solid var(--neutral) !important; outline-offset: 3px; }
p.try { font-size: 0.95rem; color: var(--muted); margin: 1rem 0 0.25rem; }
.st-key-examples { gap: 0.45rem; }

/* Result */
.result { border-top: 1px solid var(--rule); margin-top: 2.5rem; padding-top: 2.25rem; }
.verdict { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.verdict h2 {
  font-family: var(--serif); font-weight: 500; font-size: 2rem; margin: 0; padding: 0; color: var(--ink);
}
.verdict .conf { color: var(--muted); font-size: 0.98rem; }
blockquote.said {
  font-family: var(--serif); font-size: 1.5rem; line-height: 1.6; color: var(--ink) !important;
  font-style: normal; opacity: 1; border-left: 3px solid var(--c); margin: 1.5rem 0 0.75rem; padding: 0.1rem 0 0.1rem 1.25rem;
}
blockquote.said mark {
  background: linear-gradient(transparent 62%, var(--tint) 62%); color: inherit; padding: 0 0.05em;
}
p.why { color: var(--muted); font-size: 0.98rem; line-height: 1.6; margin: 0 0 2rem; }
p.why strong { color: var(--ink); font-weight: 600; }

.bars { display: grid; grid-template-columns: 5.5rem 1fr 3rem; gap: 0.7rem 1rem; align-items: center; }
.bars .name { font-size: 0.95rem; }
.bars .track { height: 6px; border-radius: 3px; background: #E2E6EB; overflow: hidden; }
.bars .fill { height: 100%; border-radius: 3px; }
.bars .pct { font-size: 0.95rem; text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }

.about { border-top: 1px solid var(--rule); margin-top: 3.5rem; padding-top: 1.5rem; }
.about p { color: var(--muted); font-size: 0.9rem; line-height: 1.6; max-width: 36rem; margin: 0; }

.site-footer {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
}
.site-footer .made { color: var(--muted); font-size: 0.9rem; margin: 0; }
.site-footer .links { display: flex; gap: 0.6rem; }
.site-footer a.profile {
  display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.45rem 0.9rem;
  border: 1px solid var(--rule); border-radius: 6px; background: var(--surface);
  color: var(--ink) !important; text-decoration: none !important; font-size: 0.9rem; font-weight: 500;
}
.site-footer a.profile:hover { border-color: var(--ink); }
.site-footer a.profile:focus-visible { outline: 2px solid var(--neutral); outline-offset: 3px; }
.site-footer a.profile svg { width: 16px; height: 16px; fill: currentColor; flex: none; }

@media (max-width: 640px) {
  .block-container { padding-top: 2.5rem; }
  h1.title { font-size: 2rem; }
  blockquote.said { font-size: 1.25rem; }
  .verdict h2 { font-size: 1.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model…")
def get_model():
    return load_model()


def use_example(text: str) -> None:
    st.session_state.text = text
    st.session_state.run = True


def highlight(text: str, words: list[str]) -> str:
    """Escape the sentence and mark the words the model leaned on."""
    out = []
    for part in re.split(r"([A-Za-z']+)", text):
        safe = html.escape(part)
        out.append(f"<mark>{safe}</mark>" if part and clean_text(part) in words else safe)
    return "".join(out)


model = get_model()

st.markdown(
    "<h1 class='title'>How does this sentence sound?</h1>"
    "<p class='lede'>Type something you might say in conversation. The model tells you whether "
    "it reads as positive, negative or neutral, and which words decided it.</p>",
    unsafe_allow_html=True,
)

st.text_area("Your sentence", key="text", height=110, label_visibility="collapsed",
             placeholder="Type a sentence…")
clicked = st.button("Check tone", type="primary")

with st.container(key="examples"):
    st.markdown("<p class='try'>Or try one of these:</p>", unsafe_allow_html=True)
    for ex in EXAMPLES:
        st.button(ex, type="tertiary", on_click=use_example, args=(ex,), key=f"ex-{ex}")

run = clicked or st.session_state.pop("run", False)
text = st.session_state.get("text", "").strip()

if run and not text:
    st.markdown("<div class='result'><p class='why'>Type a sentence above, then select "
                "<strong>Check tone</strong>.</p></div>", unsafe_allow_html=True)
elif run:
    label, scores = predict(model, text)
    words = key_words(model, text, label)
    color = f"var(--{label})"
    tint = {"positive": "rgba(30,123,85,.28)", "neutral": "rgba(79,109,143,.28)",
            "negative": "rgba(178,58,72,.26)"}[label]

    if words:
        listed = ", ".join(f"<strong>{html.escape(w)}</strong>" for w in words)
        why = f"Words that pointed to {label}: {listed}."
    elif label == "neutral":
        why = "No word in it carries much feeling, so the model reads it as neutral."
    else:
        why = "None of these words were common in the training data, so treat this as a guess."
    if scores[label] < 0.5:
        why += " The model is unsure about this one."

    bars = "".join(
        f"<div class='name'>{name.capitalize()}</div>"
        f"<div class='track'><div class='fill' style='width:{scores[name]*100:.1f}%;"
        f"background:var(--{name})'></div></div>"
        f"<div class='pct'>{scores[name]*100:.0f}%</div>"
        for name in ORDER
    )

    st.markdown(
        f"<div class='result' style='--c:{color};--tint:{tint}'>"
        f"<div class='verdict'><h2>Reads as {label}</h2>"
        f"<span class='conf'>{scores[label]*100:.0f}% confidence</span></div>"
        f"<blockquote class='said'>{highlight(text, words)}</blockquote>"
        f"<p class='why'>{why}</p>"
        f"<div class='bars'>{bars}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='about'><p>The model was trained on about 11,000 lines of everyday English "
    "conversation from the DailyDialog dataset and gets roughly 4 in 5 test sentences right. "
    "It looks at individual words rather than meaning, so sarcasm and phrases like "
    "“not bad” can fool it.</p></div>",
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/rishilrsk"
LINKEDIN_URL = "https://www.linkedin.com/in/rishisivakesh/"
GITHUB_ICON = (
    "<svg viewBox='0 0 16 16' aria-hidden='true'><path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 "
    "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94"
    "-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87"
    ".51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
    "2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 "
    "2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 "
    "2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z'/></svg>"
)
LINKEDIN_ICON = (
    "<svg viewBox='0 0 16 16' aria-hidden='true'><path d='M0 1.15C0 .52.52 0 1.18 0h13.64C15.48 0 "
    "16 .52 16 1.15v13.7c0 .63-.52 1.15-1.18 1.15H1.18C.52 16 0 15.48 0 14.85V1.15zm4.94 12.24V6.17"
    "H2.54v7.22h2.4zM3.74 5.18c.84 0 1.36-.55 1.36-1.25-.02-.71-.52-1.25-1.35-1.25-.82 0-1.36.54-1.36"
    " 1.25 0 .7.52 1.25 1.33 1.25h.02zm4.9 8.21V9.36c0-.22.02-.43.08-.59.18-.43.57-.88 1.23-.88.87 0 "
    "1.21.66 1.21 1.63v3.87h2.4V9.25c0-2.22-1.18-3.25-2.76-3.25-1.27 0-1.84.7-2.16 1.19v.03h-.02l.02"
    "-.03V6.17h-2.4c.03.68 0 7.22 0 7.22h2.4z'/></svg>"
)

st.markdown(
    "<div class='site-footer' role='contentinfo'>"
    "<p class='made'>Built by Rishi Sivakesh</p>"
    "<nav class='links' aria-label='Profiles'>"
    f"<a class='profile' href='{GITHUB_URL}' target='_blank' rel='noopener noreferrer'>"
    f"{GITHUB_ICON}GitHub</a>"
    f"<a class='profile' href='{LINKEDIN_URL}' target='_blank' rel='noopener noreferrer'>"
    f"{LINKEDIN_ICON}LinkedIn</a>"
    "</nav></div>",
    unsafe_allow_html=True,
)
