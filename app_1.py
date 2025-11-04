"""
Dynamic AI Text Analysis - Streamlit App (single-file)

Updated to improve visibility in both light and dark Streamlit themes:
- Bar chart rendered with Plotly (adapts to light/dark theme).
- WordCloud saved with transparent background so it looks good on dark or light backgrounds.
- Matplotlib use minimized; when used, facecolor/label colors are set based on detected theme.
- Theme detection uses Streamlit option 'theme.base' when available, falls back to a simple heuristic.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

# --- Safety: ensure headless matplotlib backend BEFORE importing matplotlib or other libs that import it
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Standard libs
import re
import math
import heapq
import warnings
from typing import List
import io
warnings.filterwarnings("ignore")

# Streamlit + plotting
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# NLP
import nltk
from nltk.stem import WordNetLemmatizer

# Try to import VADER
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _VADER_POSSIBLE = True
except Exception:
    _VADER_POSSIBLE = False

# Lazy transformers loader for abstractive summaries
_TRANSFORMERS_AVAILABLE = False
_TRANSFORMERS_IMPORT_ERROR = None

def try_enable_transformers():
    global _TRANSFORMERS_AVAILABLE, _TRANSFORMERS_IMPORT_ERROR
    if _TRANSFORMERS_AVAILABLE:
        return True, None
    try:
        from transformers import pipeline, AutoTokenizer  # noqa: F401
        import torch  # noqa: F401
        _TRANSFORMERS_AVAILABLE = True
        _TRANSFORMERS_IMPORT_ERROR = None
        return True, None
    except Exception as e:
        _TRANSFORMERS_AVAILABLE = False
        _TRANSFORMERS_IMPORT_ERROR = e
        err_str = str(e)
        if "Keras 3" in err_str or "Your currently installed version of Keras" in err_str or "tf-keras" in err_str:
            hint = "Keras 3 incompatibility detected. Install tf-keras (pip install tf-keras) and restart the app."
            return False, f"{err_str}\n\n{hint}"
        return False, err_str

# Ensure NLTK resources
nltk_packages = ["punkt", "wordnet", "omw-1.4", "vader_lexicon", "stopwords"]
for pkg in nltk_packages:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "vader_lexicon":
            nltk.data.find("sentiment/vader_lexicon")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()
try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# Initialize VADER if available
if _VADER_POSSIBLE:
    try:
        SIA = SentimentIntensityAnalyzer()
        _VADER_AVAILABLE = True
    except Exception:
        SIA = None
        _VADER_AVAILABLE = False
else:
    SIA = None
    _VADER_AVAILABLE = False

# Text utilities
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    toks = [w for w in t.split() if w and w not in STOPWORDS]
    toks = [lemmatizer.lemmatize(w) for w in toks]
    return " ".join(toks)

# Extractive summarizer (rule-based)
def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, sent in enumerate(sentences):
        s = sum(freq.get(w, 0) for w in word_tokens(sent))
        scores.append((s, i, sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    reduced = " ".join([s for (_score, _i, s) in top_sorted])
    return reduced

# Abstractive summarizer (lazy)
def make_abstractive_pipeline(model_name: str = "t5-small"):
    avail, err = try_enable_transformers()
    if not avail:
        raise RuntimeError(err or "transformers/torch not available")
    from transformers import pipeline  # local import
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    avail, err = try_enable_transformers()
    if not avail:
        return text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer, "model_max_length", 512) or 512
    if model_max > 16384:
        model_max = 1024
    budget = max(64, int(model_max * fraction_of_model_max))
    sentences = split_sentences(text)
    if not sentences:
        return text
    def token_count(s: str) -> int:
        ids = tokenizer.encode(s, add_special_tokens=False, truncation=False)
        return len(ids)
    joined = " ".join(sentences)
    if token_count(joined) <= budget:
        return joined
    left = 0
    right = len(sentences) - 1
    while left <= right:
        candidate = sentences[:left + 1] + sentences[right:]
        if token_count(" ".join(candidate)) <= budget:
            return " ".join(candidate)
        right -= 1
        if right < left:
            break
    first = sentences[0]
    ids = tokenizer.encode(first, add_special_tokens=False)
    ids = ids[:max(1, budget)]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def abstractive_summarize_text(text: str, model_name: str = "t5-small", max_length: int = 120, min_length: int = 20, use_reduced: bool = True) -> str:
    avail, err = try_enable_transformers()
    if not avail:
        raise RuntimeError(err or "transformers not available")
    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=6) if use_reduced else text
    trimmed = trim_for_model(reduced, model_name)
    summarizer = make_abstractive_pipeline(model_name)
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and out:
        return out[0].get("summary_text", "").strip()
    return str(out)

# Sentence-level sentiment labeling
def sentiment_label_for_sentence(sent: str) -> str:
    if SIA is not None:
        sc = SIA.polarity_scores(sent)
        compound = sc.get("compound", 0.0)
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    else:
        pos_words = {"good","great","happy","love","excellent","fantastic","amazing","best","wonderful","satisfied","pleasant"}
        neg_words = {"bad","terrible","hate","awful","worst","poor","disappointed","problem","slow","broken","angry"}
        words = set(w.lower() for w in re.findall(r"\w+", sent))
        p = len(words & pos_words)
        n = len(words & neg_words)
        if p > n:
            return "positive"
        if n > p:
            return "negative"
        return "neutral"

# --- Theme detection helper
def detect_theme():
    """
    Attempt to detect Streamlit theme: returns 'dark' or 'light'.
    Uses st.get_option('theme.base') if available; otherwise falls back to light.
    """
    try:
        base = st.get_option("theme.base")  # 'light' or 'dark'
        if base and base.lower().startswith("dark"):
            return "dark"
        return "light"
    except Exception:
        # Fallback: assume light
        return "light"

# --- Streamlit UI
st.set_page_config(page_title="Dynamic AI Text Analysis", layout="centered")
st.title("Dynamic AI Text Analysis — Sentiment, WordCloud & Summaries")
st.markdown(
    "Paste text or upload a plain text (.txt) file. The app will:\n\n"
    "- Label each sentence as positive / neutral / negative.\n"
    "- Display a bar chart with counts of positive, neutral, and negative sentences (theme-aware).\n"
    "- Generate a WordCloud (transparent background so it works on dark or light themes).\n"
    "- Provide an extractive summary and an optional abstractive (generative) summary.\n"
)

# Input: text area or upload text file
st.subheader("Input")
col_a, col_b = st.columns([3,1])
with col_a:
    text_input = st.text_area("Paste your text here", height=260, placeholder="Type or paste text to analyze...")
with col_b:
    uploaded = st.file_uploader("Or upload a plain .txt file", type=["txt"])
    st.markdown(" ")
    st.info("Provide text only; no CSV files are required.")

if uploaded is not None:
    try:
        raw = uploaded.read()
        try:
            file_text = raw.decode("utf-8")
        except Exception:
            file_text = raw.decode("latin-1")
        text_input = file_text
        st.success("Loaded text file")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

# Controls
st.subheader("Options")
col1, col2 = st.columns(2)
with col1:
    ratio = st.slider("Extractive summary ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
with col2:
    abstractive_opt = st.checkbox("Generate abstractive summary if available", value=False)
    if abstractive_opt:
        abstr_model = st.selectbox("Abstractive model", ["t5-small", "t5-base"], index=0)
    else:
        abstr_model = None

run = st.button("Analyze")

# Run analysis
if run:
    if not text_input or not text_input.strip():
        st.error("Please paste text or upload a .txt file to analyze.")
    else:
        theme = detect_theme()
        with st.spinner("Analyzing text..."):
            sentences = split_sentences(text_input)
            labels = [sentiment_label_for_sentence(s) for s in sentences]
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1

            # WordCloud: use RGBA + transparent background so it looks good in dark mode
            wc_text = clean_text(text_input)
            if not wc_text.strip():
                wc_text = "empty"
            wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap="viridis").generate(wc_text)

        # Display input (collapsed)
        st.subheader("Input text")
        st.write(text_input)

        # Sentiment bar chart — use Plotly which adapts to theme
        st.subheader("Sentiment overview")
        df_counts = {"sentiment": ["positive", "neutral", "negative"], "count": [counts["positive"], counts["neutral"], counts["negative"]]}
        import pandas as _pd
        dfc = _pd.DataFrame(df_counts)

        # choose Plotly template based on detected theme
        if theme == "dark":
            template = "plotly_dark"
            text_color = "white"
        else:
            template = "plotly_white"
            text_color = "black"

        # Colors that show well on both themes (high contrast)
        colors = {"positive": "#2b8cbe", "neutral": "#6c757d", "negative": "#f03b20"}
        fig = px.bar(dfc, x="sentiment", y="count", color="sentiment",
                     color_discrete_map=colors,
                     template=template,
                     labels={"count":"Number of sentences", "sentiment":""})
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(tickfont=dict(color=text_color))
        fig.update_yaxes(tickfont=dict(color=text_color), titlefont=dict(color=text_color))
        st.plotly_chart(fig, use_container_width=True)

        # WordCloud: render to PNG with transparent background
        st.subheader("Word Cloud")
        img_buf = io.BytesIO()
        plt.figure(figsize=(10,4), facecolor="none")
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format="png", bbox_inches="tight", transparent=True)
        plt.close()
        img_buf.seek(0)
        st.image(img_buf, use_column_width=True)

        # Summaries
        st.subheader("Summaries")
        try:
            ext = extractive_reduce(text_input, ratio=ratio)
            st.markdown("**Extractive summary**")
            st.write(ext)
        except Exception as e:
            st.error(f"Extractive summary failed: {e}")

        if abstractive_opt:
            st.markdown("**Abstractive summary**")
            avail, err = try_enable_transformers()
            if not avail:
                st.error("Abstractive summarization unavailable: " + (err or "transformers/torch not installed."))
                st.info("To enable abstractive summaries, install: pip install transformers torch sentencepiece tf-keras")
            else:
                with st.spinner("Generating abstractive summary (may take time)..."):
                    try:
                        abstr = abstractive_summarize_text(text_input, model_name=abstr_model, max_length=120, min_length=20, use_reduced=True)
                        st.write(abstr)
                    except Exception as e:
                        st.error(f"Abstractive summarization failed at runtime: {e}")
                        st.info("If the error mentions Keras 3 compatibility, run: pip install tf-keras and restart the app.")

st.markdown("---")
st.caption("Theme-aware visuals: the bar chart uses Plotly which adapts to Streamlit light/dark themes; the WordCloud image has a transparent background so it displays correctly on both.")
