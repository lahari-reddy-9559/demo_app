"""
Dynamic AI Text Analysis - Streamlit App (single-file)

Place this file in a project folder and run:
    pip install -r requirements.txt
    streamlit run app.py

This version forces a headless matplotlib backend (Agg) and reduces TF log spam
BEFORE importing matplotlib so it runs safely on headless servers (Streamlit Cloud, Docker, CI).
It includes:
 - CSV upload (expects a 'text' column) or demo dataset
 - Text cleaning (NLTK lemmatization + stopwords)
 - TF-IDF + LDA topic modeling with word clouds and bar charts
 - Extractive summarization and optional abstractive (if transformers installed)
 - Optional basic sentiment classifiers (SGD, RandomForest, optional TF NN)
 - Export topic summary CSV
"""

# SAFETY: Force headless matplotlib backend and reduce TF logs BEFORE any matplotlib import
import os
os.environ["MPLBACKEND"] = "Agg"            # safe, headless backend
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # limit TensorFlow logs

# Standard / visualization / app
import io
import re
import math
import heapq
import warnings
from typing import List
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP / ML
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Optional: TensorFlow for NN
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

# Optional: transformers for abstractive summarization
try:
    from transformers import pipeline, AutoTokenizer
    import torch
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# Ensure NLTK resources (download if missing)
nltk_packages = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
for pkg in nltk_packages:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

# Text cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Simple extractive summarizer
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS_SUM = {"a","an","the","and","or","but","if","while","with","without","to","from","by","for","of","on","in","at","is","are","was","were","this","that","these","those","it","its","be","as","which","not"}

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in STOPWORDS_SUM]

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 8) -> str:
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

# Abstractive summarization helpers (optional)
def make_abstractive_pipeline(model_name: str = "t5-small"):
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers/torch not installed.")
    device = 0 if torch and torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    if not _TRANSFORMERS_AVAILABLE:
        return text
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

def abstractive_summarize_text(text: str, model_name: str = "t5-small", max_length: int = 120, min_length: int = 20, use_extractive_reduced: bool = True) -> str:
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed.")
    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=8) if use_extractive_reduced else text
    trimmed = trim_for_model(reduced, model_name)
    summarizer = make_abstractive_pipeline(model_name)
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and out:
        return out[0].get("summary_text", "").strip()
    return str(out)

# Streamlit UI
st.set_page_config(page_title="Dynamic AI Text Analysis", layout="wide")
st.title("Dynamic AI Text Analysis")
st.markdown("Upload a CSV with a 'text' column, or use demo data. Topic modeling, summarization, sentiment modeling, and visualizations are included.")

# Sidebar options
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV with a 'text' column (optional)", type=["csv"])
use_sample = st.sidebar.checkbox("Use demo dataset", value=True)
n_topics = st.sidebar.slider("Number of LDA topics", min_value=2, max_value=12, value=5)
max_features = st.sidebar.slider("TF-IDF max features", min_value=500, max_value=10000, value=3000, step=500)
show_abstractive = st.sidebar.checkbox("Enable abstractive summarization (requires transformers)", value=False)
abstractive_model = st.sidebar.selectbox("Abstractive model", options=["t5-small", "t5-base"], index=0)
run_topic = st.sidebar.checkbox("Run topic modeling & visuals", value=True)
run_sentiment_models = st.sidebar.checkbox("Train sentiment classifiers (if sentiment labels exist)", value=False)

# Demo dataset generator
def load_demo_df(n=100):
    texts = [
        "The product arrived on time and works great. I'm happy with the purchase.",
        "I had a terrible customer service experience. They never responded to my emails.",
        "The new update improved performance and fixed many bugs.",
        "Battery life is poor and the device heats up quickly.",
        "Fantastic camera quality and amazing low-light performance.",
        "Shipping was delayed and packaging was damaged.",
        "Great value for money. Highly recommend to friends and family.",
        "App crashes on launch after the recent update. Unusable.",
        "Customer support was very helpful and solved the issue quickly.",
        "The UI is clean, responsive, and easy to use.",
    ]
    df = pd.DataFrame({"text": texts * (n // len(texts) + 1)})
    df = df.head(n).reset_index(drop=True)
    sentiments = ["positive","negative","positive","negative","positive","negative","positive","negative","positive","positive"]
    df["sentiment"] = [sentiments[i % len(sentiments)] for i in range(len(df))]
    return df

# Load data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding="latin-1")
    st.success(f"Loaded {len(df)} rows from uploaded CSV")
else:
    if use_sample:
        df = load_demo_df(n=100)
        st.info("Using demo dataset.")
    else:
        df = pd.DataFrame({"text": []})
        st.warning("No data loaded. Upload a CSV or enable demo.")

if "text" not in df.columns:
    st.error("Input data must contain a 'text' column.")
    st.stop()

df = df.dropna(subset=["text"]).reset_index(drop=True)
st.subheader("Data preview")
st.dataframe(df.head())

with st.spinner("Cleaning text..."):
    df["cleaned_text"] = df["text"].astype(str).apply(clean_text)
    try:
        df["tokens"] = df["cleaned_text"].apply(lambda x: word_tokenize(x))
    except Exception:
        df["tokens"] = df["cleaned_text"].apply(lambda x: x.split())

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
if len(df) == 0:
    st.error("No rows to process.")
    st.stop()

tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

# Topic modeling
if run_topic:
    st.header("Topic Modeling")
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=42)
    with st.spinner("Fitting LDA..."):
        lda.fit(tfidf_matrix)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    def top_words_for_topic(topic_id, model, vectorizer, top_n=10):
        fn = vectorizer.get_feature_names_out()
        topic = model.components_[topic_id]
        top_idx = topic.argsort()[-top_n:][::-1]
        return [fn[i] for i in top_idx]

    st.subheader("Top words per topic")
    cols = st.columns(min(n_topics, 5))
    for t in range(n_topics):
        words = top_words_for_topic(t, lda, tfidf_vectorizer, top_n=10)
        with cols[t % len(cols)]:
            st.markdown(f"**Topic {t+1}**")
            st.write(", ".join(words))

    topic_dists = lda.transform(tfidf_matrix)
    topics = np.argmax(topic_dists, axis=1)
    df["topic"] = topics
    df["topic_prob"] = topic_dists.max(axis=1)

    st.subheader("Documents per topic")
    topic_counts = df["topic"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=[f"Topic {i}" for i in topic_counts.index], y=topic_counts.values, palette="Spectral", ax=ax)
    ax.set_ylabel("Document count")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Word Clouds per topic")
    wc_cols = st.columns(min(3, n_topics))
    for t in range(n_topics):
        topic = lda.components_[t]
        top_idx = topic.argsort()[-100:][::-1]
        freqs = {feature_names[i]: float(topic[i]) for i in top_idx}
        wc = WordCloud(width=400, height=200, background_color="white", colormap="tab10").generate_from_frequencies(freqs)
        c = wc_cols[t % len(wc_cols)]
        with c:
            st.image(wc.to_array(), use_column_width=True, caption=f"Topic {t+1}")

# Sentiment modeling if available and requested
if "sentiment" in df.columns and run_sentiment_models:
    st.header("Sentiment Modeling & Integration")
    df = df.dropna(subset=["sentiment"]).reset_index(drop=True)
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["sentiment"].astype(str))
    X_train, X_val, y_train, y_val = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    st.subheader("SGD (Logistic) quick training")
    clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=0, learning_rate="adaptive", eta0=0.01)
    classes = np.unique(y_train)
    n_epochs = 10
    train_losses_lr = []
    val_losses_lr = []
    for epoch in range(n_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        clf.partial_fit(X_shuf, y_shuf, classes=classes)
        prob_train = clf.predict_proba(X_train)
        prob_val = clf.predict_proba(X_val)
        train_losses_lr.append(log_loss(y_train, prob_train))
        val_losses_lr.append(log_loss(y_val, prob_val))
    fig, ax = plt.subplots()
    ax.plot(range(1, n_epochs+1), train_losses_lr, label="train log loss")
    ax.plot(range(1, n_epochs+1), val_losses_lr, label="val log loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.legend()
    st.pyplot(fig)

    st.subheader("RandomForest quick eval")
    rf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    st.write(f"RandomForest accuracy (n_estimators=50): {acc_rf:.4f}")

    if _TF_AVAILABLE:
        st.subheader("Simple NN (TensorFlow)")
        model = Sequential([
            Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(len(np.unique(y)), activation="softmax")
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_data=(X_val.toarray(), y_val), verbose=0)
        fig, ax = plt.subplots()
        ax.plot(hist.history["loss"], label="train loss")
        ax.plot(hist.history["val_loss"], label="val loss")
        ax.legend()
        st.pyplot(fig)
        loss, acc = model.evaluate(X_val.toarray(), y_val, verbose=0)
        st.write(f"NN accuracy: {acc:.4f}")
    else:
        st.info("TensorFlow not available - skip NN training. Install tensorflow to enable.")

    # Sentiment per topic
    if "topic" in df.columns:
        st.subheader("Sentiment distribution per topic")
        df_temp = df.copy()
        df_temp["sent_label"] = label_enc.inverse_transform(y)
        stack_df = df_temp.groupby(["topic", "sent_label"]).size().unstack(fill_value=0)
        st.write(stack_df)
        fig, ax = plt.subplots(figsize=(10,5))
        stack_df_norm = stack_df.div(stack_df.sum(axis=1), axis=0).fillna(0)
        stack_df_norm.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_ylabel("Fraction of docs")
        ax.set_xlabel("Topic")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Summarization UI
st.header("Summarization")
input_mode = st.radio("Input mode", ["Text input", "Use a document from dataset"])
if input_mode == "Text input":
    text_input = st.text_area("Enter text to summarize", height=200)
else:
    idx = st.number_input("Document index (0-based)", min_value=0, max_value=max(0, len(df)-1), value=0)
    text_input = df.loc[int(idx), "text"] if len(df) > 0 else ""

st.markdown("Summarization options")
ratio = st.slider("Extractive ratio", min_value=0.1, max_value=1.0, value=0.3)
do_extractive = st.checkbox("Do extractive summarization", value=True)
do_abstractive = st.checkbox("Do abstractive summarization", value=show_abstractive and _TRANSFORMERS_AVAILABLE)
ab_model = st.selectbox("Abstractive model", ["t5-small", "t5-base"], index=0) if do_abstractive else None
max_len = st.slider("Abstractive max tokens", min_value=32, max_value=512, value=120)

if st.button("Generate summary"):
    if not text_input or not text_input.strip():
        st.error("Please provide text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            if do_extractive:
                ext = extractive_reduce(text_input, ratio=ratio)
                st.subheader("Extractive Summary")
                st.write(ext)
            if do_abstractive:
                if not _TRANSFORMERS_AVAILABLE:
                    st.error("Transformers not installed; abstractive unavailable.")
                else:
                    try:
                        abstr = abstractive_summarize_text(text_input, model_name=ab_model, max_length=max_len, min_length=20, use_extractive_reduced=True)
                        st.subheader("Abstractive Summary")
                        st.write(abstr)
                    except Exception as e:
                        st.error(f"Abstractive summarization failed: {e}")

# Export topic summary
if "topic" in df.columns:
    st.subheader("Export Topic Summary")
    try:
        df_int = df.copy()
        if "sentiment" in df_int.columns:
            sent_map_signed = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
            df_int["sent_signed"] = df_int["sentiment"].map(lambda s: sent_map_signed.get(s, 0.0))
        else:
            df_int["sent_signed"] = 0.0
        agg = df_int.groupby("topic").agg(n_docs=("text","count"), mean_sent_signed=("sent_signed","mean")).reset_index()
        agg["top_words"] = agg["topic"].apply(lambda t: ", ".join(top_words_for_topic(int(t), lda, tfidf_vectorizer, top_n=10)))
        csv = agg.to_csv(index=False).encode("utf-8")
        st.download_button("Download topic summary CSV", data=csv, file_name="lda_topic_sentiment_summary.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to build export: {e}")

st.markdown("---")
st.caption("Designed with light, color-blind-friendly palettes where possible.")

if st.checkbox("Show recommended pip install line"):
    st.code("""pip install streamlit pandas numpy scikit-learn nltk matplotlib seaborn wordcloud gensim
# Optional (for abstractive summaries & NN):
pip install transformers torch sentencepiece tensorflow rouge-score""")
