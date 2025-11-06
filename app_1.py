"""
Text Overanalysis Engine - Streamlit App (Sarcastic Edition)

Focus: Minimal detail, unique, sarcastic UI style, dark/light mode compatible.
Functionality: Sentiment, Word Cloud, Extractive/Generative Summaries.
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
import pandas as pd
import altair as alt
warnings.filterwarnings("ignore")

# Streamlit + plotting
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

# NLP
import nltk
from nltk.stem import WordNetLemmatizer

# --- Dependency Setup (Kept Minimal) ---
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    SIA = SentimentIntensityAnalyzer()
except Exception:
    SIA = None

_TRANSFORMERS_AVAILABLE = False
_TRANSFORMERS_IMPORT_ERROR = None

def try_enable_transformers():
    """Checks for transformers and returns a brief, sarcastic error message if missing."""
    global _TRANSFORMERS_AVAILABLE, _TRANSFORMERS_IMPORT_ERROR
    if _TRANSFORMERS_AVAILABLE: return True, None
    try:
        from transformers import pipeline, AutoTokenizer
        import torch
        _TRANSFORMERS_AVAILABLE = True
        _TRANSFORMERS_IMPORT_ERROR = None
        return True, None
    except Exception as e:
        _TRANSFORMERS_AVAILABLE = False
        err_str = str(e)
        
        # Sarcastic, brief error message
        if "No module named 'transformers'" in err_str:
             return False, "You forgot to install the big libraries. Run: `pip install transformers torch sentencepiece tf-keras`"
        if "Keras 3" in err_str:
            return False, "Keras is having a mid-life crisis. Try: `pip install tf-keras`"
        return False, f"Some Python nonsense broke it. ({err_str[:40]}...)"


# Ensure NLTK resources (silent download)
for pkg in ["punkt", "wordnet", "omw-1.4", "vader_lexicon", "stopwords"]:
    try:
        if pkg == "punkt": nltk.data.find("tokenizers/punkt")
        elif pkg == "vader_lexicon": nltk.data.find("sentiment/vader_lexicon")
        else: nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True) 
        except Exception: pass

lemmatizer = WordNetLemmatizer()
try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# --- Core Logic Functions (Preserved) ---
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    toks = [lemmatizer.lemmatize(w) for w in t.split() if w and w not in STOPWORDS]
    return " ".join(toks)

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1: return text
    freq = {}
    for sent in sentences:
        for w in word_tokens(sent): freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, sent in enumerate(sentences):
        s = sum(freq.get(w, 0) for w in word_tokens(sent))
        scores.append((s, i, sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join([s for (_score, _i, s) in top_sorted])

@st.cache_resource
def make_abstractive_pipeline(model_name: str = "t5-small"):
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err or "models not available")
    from transformers import pipeline
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    avail, err = try_enable_transformers()
    if not avail: return text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer, "model_max_length", 512) or 1024
    budget = max(64, int(model_max * fraction_of_model_max))
    sentences = split_sentences(text)
    if not sentences: return text
    
    def token_count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False, truncation=False))
    
    joined = " ".join(sentences)
    if token_count(joined) <= budget: return joined
    
    # Simple trimming heuristic
    trimmed_sents = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = token_count(sent)
        if current_tokens + sent_tokens + 2 <= budget: # +2 for delimiters
            trimmed_sents.append(sent)
            current_tokens += sent_tokens
        elif current_tokens == 0:
            # If the first sentence is too long, truncate it
            ids = tokenizer.encode(sent, add_special_tokens=False)[:budget]
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return " ".join(trimmed_sents)

def abstractive_summarize_text(text: str, model_name: str = "t5-small", max_length: int = 120, min_length: int = 20, use_reduced: bool = True) -> str:
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err or "models not available")
    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=6) if use_reduced else text
    trimmed = trim_for_model(reduced, model_name)
    summarizer = make_abstractive_pipeline(model_name)
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and out:
        return out[0].get("summary_text", "").strip()
    return str(out)

def sentiment_label_for_sentence(sent: str) -> str:
    """ Returns one of 'positive', 'neutral', 'negative'. """
    if SIA is not None:
        sc = SIA.polarity_scores(sent)
        compound = sc.get("compound", 0.0)
        if compound >= 0.05: return "positive"
        elif compound <= -0.05: return "negative"
        else: return "neutral"
    else: # Heuristic fallback
        pos_words = {"good","great","happy","love"}
        neg_words = {"bad","terrible","hate","awful"}
        words = set(w.lower() for w in re.findall(r"\w+", sent))
        p = len(words & pos_words)
        n = len(words & neg_words)
        if p > n: return "positive"
        if n > p: return "negative"
        return "neutral"

# --- SARCASTIC UI & STYLE ---

st.set_page_config(page_title="Text Overanalysis", layout="centered")

# Minimalist CSS with dark/light mode adaptability
st.markdown(
    """
    <style>
    /* Use bold, generic font */
    html, body, [class*="st-"] {
        font-family: sans-serif !important;
    }
    .stApp { 
        background-color: var(--background-color); 
        color: var(--text-color);
    }
    /* Sarcastic result box */
    .result-box { 
        padding: 18px; 
        border-radius: 6px; 
        background: var(--secondary-background-color); 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px;
        border: 1px solid #ff4b4b; /* Subtle red border for "sarcasm" */
    }
    .stButton>button {
        background-color: #ff4b4b; /* Red button for drama */
        color: white; 
        font-weight: bold;
        border-radius: 4px;
        border: none;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🤦 Text Overanalysis Tool")
st.markdown("Honestly, it just counts words and gets feelings wrong. But sure, let's analyze it.")
st.info("⚠️ **Rules:** Put text in. Press the button. Complain later.")

# --- OPTIONS SECTION ---
st.header("1. Settings (Boring but necessary)")
col1, col2, col3 = st.columns(3)

with col1:
    ratio = st.slider("Keep-Sentence Ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    st.caption("How much summary you want to read.")

with col2:
    abstractive_opt = st.checkbox("Generate Smarter Summary (If you installed the huge libraries)", value=False)
    st.caption("Warning: Requires a functioning internet/PC.")

with col3:
    if abstractive_opt:
        abstr_model = st.selectbox("Model Overkill", ["t5-small", "t5-base"], index=0)
    else:
        abstr_model = None
        st.markdown("*Disabled: Too much work*")

# --- INPUT SECTION ---
st.markdown("---")
with st.form(key='analysis_form'):
    st.header("2. Hand Over the Text")

    col_a, col_b = st.columns([3,1])
    
    if 'default_text_input' not in st.session_state: st.session_state.default_text_input = ""

    with col_a:
        text_input = st.text_area(
            "Dump Text Here", 
            height=260, 
            placeholder="Go on, paste your masterpiece...",
            value=st.session_state.default_text_input
        )

    with col_b:
        uploaded = st.file_uploader("Or Upload .txt (Easier)", type=["txt"])
        
        if uploaded is not None:
            try:
                raw = uploaded.read()
                file_text = raw.decode("utf-8", errors='ignore')
                text_input = file_text
                st.session_state.default_text_input = file_text
                st.toast("✅ File loaded. Now hurry up and click the button.", icon='📄')
            except Exception as e:
                st.error(f"❌ File upload failed. Why is it always something? Error: {e}")
            
    # Form submission button
    run = st.form_submit_button("🛑 Stop Procrastinating, Analyze")

# --- RESULTS DISPLAY ---
st.markdown("---")

if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Seriously? I can't analyze air.")
    else:
        st.session_state.default_text_input = text_input 
        
        with st.spinner("Judging your text..."):
            sentences = split_sentences(text_input)
            labels = [sentiment_label_for_sentence(s) for s in sentences]
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for lab in labels: counts[lab] = counts.get(lab, 0) + 1
            wc_text = clean_text(text_input)
            if not wc_text.strip(): wc_text = "empty"
            wc = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(wc_text)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.header("3. The Obvious Results")
        st.markdown("---")
        
        # --- Sentiment Plot ---
        st.subheader("Sentiment: How Wrong Am I?")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ["Positive", "Neutral", "Negative"],
            'Count': [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
        })
        
        sentiment_colors = {
            "Positive": "#4daf4a", 
            "Neutral": "#6c757d",  
            "Negative": "#e41a1c"
        }
        
        base = alt.Chart(sentiment_data).encode(
            x=alt.X('Sentiment', sort=["Positive", "Neutral", "Negative"]), y='Count'
        )
        
        chart = base.mark_bar().encode(
            color=alt.Color('Sentiment', scale=alt.Scale(domain=list(sentiment_colors.keys()), 
                                                        range=list(sentiment_colors.values()))),
            tooltip=['Sentiment', 'Count']
        ).properties(title="Emotional Breakdown (Probably Inaccurate)")
        
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(sentiment_data.sort_values(by='Count', ascending=False), hide_index=True, use_container_width=True)

        # --- Word Cloud ---
        st.markdown("---")
        st.subheader("Word Cloud: The Hype Map")
        
        img_buf = io.BytesIO()
        plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation='bilinear'); plt.axis("off"); plt.tight_layout(pad=0)
        plt.savefig(img_buf, format="png", bbox_inches="tight"); plt.close()
        img_buf.seek(0)
        st.image(img_buf, use_column_width=True)
        st.caption("These are the words you used most. Stop repeating yourself.")

        # --- Summaries ---
        st.markdown("---")
        st.subheader("Summaries: Did you really need all that text?")
        
        # Extractive summary
        st.markdown("##### ✂️ Extractive Summary (The lazy way)")
        try:
            ext = extractive_reduce(text_input, ratio=ratio)
            st.success(ext)
        except Exception as e:
            st.error(f"Failed. Couldn't even copy and paste correctly. Error: {e}")

        # Abstractive summary (if requested)
        if abstractive_opt:
            st.markdown("##### 🧠 Generative Summary (The expensive way)")
            avail, err = try_enable_transformers()
            if not avail:
                st.error(f"❌ **Failed.** Reason: {err}")
            else:
                with st.spinner(f"Waiting for **{abstr_model}** to wake up..."):
                    try:
                        abstr = abstractive_summarize_text(text_input, model_name=abstr_model)
                        st.success(abstr)
                    except Exception as e:
                        st.error(f"❌ ML model choked. Maybe use a smaller text next time? Error: {e}")
                        
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built for overthinkers. Now go outside.")
