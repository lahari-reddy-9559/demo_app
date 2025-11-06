"""
Dynamic AI Text Analysis - Streamlit App (single-file)

Features:
- Accepts pasted text or a .txt file upload for analysis.
- Sentence-level sentiment labeling (VADER when available, fallback heuristic).
- Neutral, background-friendly color palette (suitable for light/white backgrounds).
- Shows a bar chart of positive / neutral / negative sentence counts (Now using st.bar_chart for animation).
- Generates a WordCloud of the input text.
- Produces an extractive summary (rule-based) and, if available, an abstractive (generative) summary
  using Hugging Face transformers loaded lazily.
- Graceful handling of transformers / Keras 3 incompatibility (suggests pip install tf-keras).
- Headless-safe (forces matplotlib Agg backend).

Run:
1. pip install -r requirements.txt
2. streamlit run app.py
"""

# Safety: ensure headless matplotlib backend BEFORE importing matplotlib or other libs that import it
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Standard libs
import re
import math
import heapq
import warnings
from typing import List
import pandas as pd # <-- Added for st.bar_chart
warnings.filterwarnings("ignore")

# Streamlit + plotting
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# NLP
import nltk
from nltk.stem import WordNetLemmatizer

# Attempt to import VADER (NLTK sentiment analyzer)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    _VADER_POSSIBLE = True
except Exception:
    _VADER_POSSIBLE = False

# Lazy transformers loader (for abstractive summarization)
_TRANSFORMERS_AVAILABLE = False
_TRANSFORMERS_IMPORT_ERROR = None

def try_enable_transformers():
    """
    Try to import transformers and torch lazily.
    Returns (available: bool, error_message: str|None).
    Detects Keras 3 incompatibility and returns a helpful hint to install tf-keras.
    """
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
            hint = (
                "Keras 3 incompatibility detected. Install the compatibility package:\n\n"
                "    pip install tf-keras\n\n"
                "Then restart the app. This lets Transformers' TF code work with Keras 3."
            )
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

# Initialize VADER if installed properly
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
@st.cache_resource
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
    """
    Returns one of 'positive', 'neutral', 'negative'.
    Uses VADER if available; otherwise uses a simple heuristic based on seed words.
    """
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

# --- UI/UX Enhancements ---

# Streamlit UI (neutral, background-friendly palette)
st.set_page_config(page_title="Dynamic AI Text Analysis", layout="centered")

# Custom CSS for a modern, animated look
st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(180deg, #f0f8ff 0%, #ffffff 100%); 
        color: #333;
    }
    .stHeader {
        color: #1a73e8; /* Google Blue */
        text-align: center;
        font-weight: 700;
        padding-top: 10px;
        padding-bottom: 5px;
    }
    /* Stylish containers for the results */
    .result-box { 
        padding: 15px; 
        border-radius: 12px; 
        background: #ffffff; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        transition: box-shadow 0.3s ease-in-out;
    }
    .result-box:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    /* Make button stand out */
    .stButton>button {
        background-color: #1a73e8; 
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1764cf;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🤖 Dynamic AI Text Analysis")
st.markdown(
    """
    **Instantly analyze text** for sentiment, keyword frequency, and generate automatic summaries.
    This app provides **sentence-level sentiment** and two types of summarization.
    """
)
st.info("💡 **Instructions:** Paste text or upload a .txt file below, adjust the summary options, and click **Analyze**.")

# Use a form for a more convenient grouped input/control area
with st.form(key='analysis_form'):
    st.header("1️⃣ Input & Options")

    # Input: text area or upload text file (side by side)
    col_a, col_b = st.columns([3,1])
    
    # Initialize text_input to a default value for the form
    if 'default_text_input' not in st.session_state:
        st.session_state.default_text_input = ""

    with col_a:
        text_input = st.text_area(
            "📝 Paste your text here (or use the file upload)", 
            height=260, 
            placeholder="Type or paste text to analyze...",
            value=st.session_state.default_text_input # Use session state for sticky text
        )

    with col_b:
        uploaded = st.file_uploader("📂 Upload a plain .txt file", type=["txt"])
        
        # Options moved inside the columns for better layout
        st.markdown("---")
        st.caption("Summary Controls:")
        ratio = st.slider("Extractive summary ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
        abstractive_opt = st.checkbox("Generate abstractive summary (AI Model)", value=False)
        if abstractive_opt:
            abstr_model = st.selectbox("Abstractive model", ["t5-small", "t5-base"], index=0)
        else:
            abstr_model = None

    # Process file upload if it exists
    if uploaded is not None:
        try:
            raw = uploaded.read()
            try:
                file_text = raw.decode("utf-8")
            except Exception:
                file_text = raw.decode("latin-1")
            
            # Overwrite text_input and update session state
            text_input = file_text
            st.session_state.default_text_input = file_text
            st.toast("✅ File loaded! Click 'Analyze'.", icon='📄')
        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")
            
    # Form submission button
    run = st.form_submit_button("✨ Analyze Text")

# --- Analysis Logic and Results Display ---

if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Please paste text or upload a .txt file to analyze.")
    else:
        # Update text input state for persistence
        st.session_state.default_text_input = text_input 
        
        with st.spinner("Processing sentiment and preparing analysis..."):
            sentences = split_sentences(text_input)
            labels = [sentiment_label_for_sentence(s) for s in sentences]
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1

            # WordCloud generation
            wc_text = clean_text(text_input)
            if not wc_text.strip():
                wc_text = "empty"
            # Using 'RdYlGn' or 'viridis' for general readability
            wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(wc_text)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.header("2️⃣ Results Overview")
        st.markdown("---")
        
        # --- Sentiment Plot (using st.bar_chart for native animation) ---
        st.subheader("📊 Sentiment Breakdown")
        st.markdown("Sentence-level counts:")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ["Positive", "Neutral", "Negative"],
            'Count': [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
        })
        
        st.bar_chart(
            sentiment_data.set_index('Sentiment'), 
            color=['#2b8cbe', '#6c757d', '#f03b20'] # Use the same neutral, accessible colors
        )
        st.dataframe(sentiment_data, hide_index=True, use_container_width=True)


        # --- Word Cloud ---
        st.markdown("---")
        st.subheader("☁️ Word Cloud")
        
        img_buf = io.BytesIO()
        plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close()
        img_buf.seek(0)
        st.image(img_buf, use_column_width=True)
        
        # --- Summaries ---
        st.markdown("---")
        st.subheader("📑 Summaries")
        
        # Extractive summary
        st.markdown("##### **Rule-Based (Extractive) Summary**")
        try:
            ext = extractive_reduce(text_input, ratio=ratio)
            st.success(ext)
        except Exception as e:
            st.error(f"Extractive summary failed: {e}")

        # Abstractive summary (if requested)
        if abstractive_opt:
            st.markdown("##### **Generative (Abstractive) Summary**")
            avail, err = try_enable_transformers()
            if not avail:
                st.error("❌ Abstractive summarization unavailable: " + (err or "transformers/torch not installed."))
                st.info("To enable abstractive summaries, install: `pip install transformers torch sentencepiece tf-keras`")
            else:
                with st.spinner(f"Generating abstractive summary using **{abstr_model}** (may take a moment for download/computation)..."):
                    try:
                        abstr = abstractive_summarize_text(text_input, model_name=abstr_model, max_length=120, min_length=20, use_reduced=True)
                        st.success(abstr)
                    except Exception as e:
                        st.error(f"❌ Abstractive summarization failed at runtime: {e}")
                        st.info("If the error mentions Keras 3 compatibility, run: `pip install tf-keras` and restart the app.")
        
        st.markdown('</div>', unsafe_allow_html=True) # Close the result-box div

st.markdown("---")
st.caption("Enjoy the service. Your feedback is valuable! Thank you for visiting our page.")
