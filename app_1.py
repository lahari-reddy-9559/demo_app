"""
Dynamic AI Text Analysis - Streamlit App (single-file)

Features:
- Professional, cross-mode compatible UI/UX with smooth Altair charting.
- Input via pasted text or .txt file upload.
- Sentence-level sentiment (VADER/heuristic fallback).
- WordCloud generation with a neutral, clear color scheme.
- Rule-based (extractive) and optional ML-based (abstractive) summarization.
- All model options (like t5-base) are available before the 'Analyze' button is pressed.
- Robust dependency checking for 'transformers'.

Run:
1. pip install -r requirements.txt (Ensure this includes pandas, streamlit, nltk, wordcloud, altair, and the required ML libs if needed)
2. streamlit run app.py
"""

# Safety: ensure headless matplotlib backend BEFORE importing matplotlib or other libs that import it
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import transformers 
import torch 
import sentencepiece 
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
        
        # Specific Keras 3 check
        if "Keras 3" in err_str or "tf-keras" in err_str:
            hint = (
                "Keras 3 incompatibility detected. Please install the compatibility package:\n\n"
                "    `pip install tf-keras`\n\n"
                "Then restart the application. This ensures Transformers' TF code works with Keras 3."
            )
            return False, f"Keras/TensorFlow Error: {err_str}\n\n{hint}"
        
        # General missing module check
        if "No module named 'transformers'" in err_str or "'transformers'" in err_str:
             return False, "Core ML Library Missing. To enable generative summaries, please run: `pip install transformers torch sentencepiece tf-keras`"

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

# Text utilities (functions remain identical as per user request to maintain functionality)
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

def sentiment_label_for_sentence(sent: str) -> str:
    """ Returns one of 'positive', 'neutral', 'negative'. """
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

# --- UI/UX Enhancements and Dark Mode Fixes ---

st.set_page_config(page_title="Text Analysis Dashboard", layout="centered")

# Custom CSS for dark mode compatibility and refined aesthetic
st.markdown(
    """
    <style>
    /* Ensure colors adapt to Streamlit's theme */
    .stApp { 
        background-color: var(--background-color); 
        color: var(--text-color);
    }
    /* Styling for the main result container */
    .result-box { 
        padding: 18px; 
        border-radius: 10px; 
        background: var(--secondary-background-color); 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(128, 128, 128, 0.1); 
        margin-bottom: 25px;
        transition: box-shadow 0.3s ease-in-out;
    }
    .result-box:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    /* Button style remains bold and distinct */
    .stButton>button {
        background-color: #1a73e8; 
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("💡 Text Insights Engine")
st.markdown(
    """
    Analyze textual data using a combination of fast **rule-based methods** (Extractive Summary, VADER Sentiment) 
    and optional **Transformer models** for deeper, generative insights (Abstractive Summary).
    """
)
st.info("📌 **Get Started:** Input your text, configure the summary options below, and run the analysis.")

# --- OPTIONS SECTION (Immediate selection) ---
st.header("1. Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    ratio = st.slider("Extractive Summary Length Ratio", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    st.caption("Controls the proportion of sentences kept for the rule-based summary.")

with col2:
    abstractive_opt = st.checkbox("Enable Generative Summary (Requires ML Libraries)", value=False)
    st.caption("Check this to use models like T5 for abstractive summarization.")

with col3:
    if abstractive_opt:
        abstr_model = st.selectbox("Generative Model Selection", ["t5-small", "t5-base"], index=0)
    else:
        abstr_model = None
        st.markdown("— *Disabled* —")

# --- INPUT SECTION (Inside the Form) ---
st.markdown("---")
with st.form(key='analysis_form'):
    st.header("2. Input Source")

    col_a, col_b = st.columns([3,1])
    
    if 'default_text_input' not in st.session_state:
        st.session_state.default_text_input = ""

    with col_a:
        text_input = st.text_area(
            "Paste Text Here", 
            height=260, 
            placeholder="Enter the document or article text for analysis...",
            value=st.session_state.default_text_input
        )

    with col_b:
        uploaded = st.file_uploader("Upload .txt File", type=["txt"])
        st.markdown("---")
        
        if uploaded is not None:
            try:
                raw = uploaded.read()
                try:
                    file_text = raw.decode("utf-8")
                except Exception:
                    file_text = raw.decode("latin-1")
                
                text_input = file_text
                st.session_state.default_text_input = file_text
                st.toast("✅ File loaded successfully! Click 'Run Analysis'.", icon='📄')
            except Exception as e:
                st.error(f"❌ Failed to read uploaded file: {e}")
            
    # Form submission button
    run = st.form_submit_button("🚀 Run Analysis")

# --- Analysis Logic and Results Display ---
st.markdown("---")

if run:
    if not text_input or not text_input.strip():
        st.error("🚨 Please provide text in the input area or upload a file to begin analysis.")
    else:
        st.session_state.default_text_input = text_input 
        
        with st.spinner("Processing document..."):
            sentences = split_sentences(text_input)
            labels = [sentiment_label_for_sentence(s) for s in sentences]
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1

            # WordCloud generation
            wc_text = clean_text(text_input)
            if not wc_text.strip():
                wc_text = "empty"
            # Using 'Dark2' colormap for a professional, cross-mode palette
            wc = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(wc_text)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.header("3. Analysis Results")
        st.markdown("---")
        
        # --- Sentiment Plot (Altair for cross-mode coloring) ---
        st.subheader("Emotion and Polarity Distribution")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ["Positive", "Neutral", "Negative"],
            'Count': [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
        })
        
        # Define the colors: professional blue/gray/red palette
        sentiment_colors = {
            "Positive": "#4daf4a", # Green for Positive
            "Neutral": "#6c757d",  # Gray for Neutral
            "Negative": "#e41a1c"  # Red for Negative
        }
        
        base = alt.Chart(sentiment_data).encode(
            x=alt.X('Sentiment', sort=["Positive", "Neutral", "Negative"]),
            y='Count'
        )
        
        chart = base.mark_bar().encode(
            color=alt.Color('Sentiment', scale=alt.Scale(domain=list(sentiment_colors.keys()), 
                                                        range=list(sentiment_colors.values()))),
            tooltip=['Sentiment', 'Count']
        ).properties(
            title="Sentence Counts by Sentiment Category"
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(sentiment_data.sort_values(by='Count', ascending=False), hide_index=True, use_container_width=True)


        # --- Word Cloud ---
        st.markdown("---")
        st.subheader("Key Topic Visualization (Word Cloud)")
        
        col_wc_img, col_wc_info = st.columns([3, 1])
        
        img_buf = io.BytesIO()
        plt.figure(figsize=(10,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close()
        img_buf.seek(0)
        
        with col_wc_img:
            st.image(img_buf, use_column_width=True)
        
        with col_wc_info:
            st.markdown("The Word Cloud highlights **frequently occurring words** in the document after removing common stop words (like 'the', 'is', 'a').")
            st.caption(f"Total Sentences: {len(sentences)}")
            st.caption(f"Analyzed Tokens: {len(wc_text.split())}")

        
        # --- Summaries ---
        st.markdown("---")
        st.subheader("Automated Summarization")
        
        # Extractive summary
        st.markdown("##### 📝 Extractive Summary (Rule-Based)")
        try:
            ext = extractive_reduce(text_input, ratio=ratio)
            st.success(ext)
        except Exception as e:
            st.error(f"Extractive summary failed: {e}")

        # Abstractive summary (if requested)
        if abstractive_opt:
            st.markdown("##### ✨ Generative Summary (AI Model)")
            avail, err = try_enable_transformers()
            if not avail:
                st.error(f"❌ **Generative Summary Unavailable**\n\n**Reason:** {err}")
            else:
                with st.spinner(f"Generating summary using **{abstr_model}**... (This step may require model download/initialization.)"):
                    try:
                        abstr = abstractive_summarize_text(text_input, model_name=abstr_model, max_length=120, min_length=20, use_reduced=True)
                        st.success(abstr)
                    except Exception as e:
                        st.error(f"❌ Generative summarization failed at runtime: {e}")
                        
        st.markdown('</div>', unsafe_allow_html=True) # Close the result-box div

st.markdown("---")
st.caption("Powered by Streamlit, NLTK, and Hugging Face Transformers. Thank you for using this analysis tool.")



