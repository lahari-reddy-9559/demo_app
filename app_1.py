#!/usr/bin/env python3
"""
Combined FastAPI backend + Streamlit frontend in one file.

Usage:
  - Run backend only:
      python app_combined.py --mode backend --host 0.0.0.0 --port 8000
  - Run frontend only (assumes backend is running at BACKEND_URL):
      python app_combined.py --mode frontend --backend-url http://localhost:8000
  - Run both together (starts backend in-thread, launches Streamlit as subprocess):
      python app_combined.py --mode both --host 0.0.0.0 --port 8000

Notes:
  - This is intended for local development/demo only.
  - Requires packages: fastapi, uvicorn, streamlit, httpx, scikit-learn, gensim, nltk, transformers (optional), torch (optional)
"""

import argparse
import os
import sys
import threading
import time
import subprocess
import logging
from typing import List, Optional, Dict, Any

# ---------------------------
# Backend imports & logic
# ---------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline as hf_pipeline  # optional; import error handled below

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("combined_app")

# Ensure NLTK resources (quietly)
_nltk_needed = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
for pkg in _nltk_needed:
    try:
        nltk.data.find(pkg)
    except Exception:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 8) -> str:
    import re, heapq, math
    _SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
    def split_sentences(text): return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    def word_tokens(text):
        return [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in stop_words]
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

# In-memory state for simple lazy models
STATE = {
    "tfidf_vectorizer": None,
    "lda_model": None,
    "lda_trained_on": None,
    "abstractive_pipeline": None,
}

def train_lda_on_texts(texts: List[str], n_components: int = 5, max_iter: int = 10):
    vec = TfidfVectorizer(max_features=5000)
    cleaned = [clean_text(t) for t in texts]
    X = vec.fit_transform(cleaned)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, random_state=42)
    lda.fit(X)
    STATE["tfidf_vectorizer"] = vec
    STATE["lda_model"] = lda
    STATE["lda_trained_on"] = cleaned
    return lda, vec, X

def get_abstractive_pipeline():
    if STATE["abstractive_pipeline"] is None:
        try:
            logger.info("Loading abstractive summarization pipeline (t5-small). This may take time...")
            STATE["abstractive_pipeline"] = hf_pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)
        except Exception as e:
            logger.warning(f"Could not load transformers summarization pipeline: {e}")
            STATE["abstractive_pipeline"] = None
    return STATE["abstractive_pipeline"]

# FastAPI app
backend_app = FastAPI(title="Combined Dynamic Text Analysis Backend")

class AnalyzeOptions(BaseModel):
    max_topics: Optional[int] = 8
    use_abstractive: Optional[bool] = True

class AnalyzeRequest(BaseModel):
    texts: List[str]
    mode: Optional[str] = "batch"
    options: Optional[AnalyzeOptions] = AnalyzeOptions()

@backend_app.get("/health")
def health():
    return {"status": "ok", "models_loaded": {k: (v is not None) for k, v in STATE.items()}}

@backend_app.post("/analyze")
def analyze(req: AnalyzeRequest):
    texts = req.texts
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    options = req.options or AnalyzeOptions()
    max_topics = int(options.max_topics or 8)
    use_abstractive = bool(options.use_abstractive)

    cleaned_texts = [clean_text(t) for t in texts]
    tokenized_texts = [word_tokenize(ct) for ct in cleaned_texts]

    # Train or reuse LDA
    if STATE["lda_model"] is None or STATE["lda_trained_on"] is None:
        lda_model, vec, X = train_lda_on_texts(cleaned_texts, n_components=max(2, max_topics), max_iter=10)
    else:
        lda_model = STATE["lda_model"]
        vec = STATE["tfidf_vectorizer"]
        try:
            X = vec.transform(cleaned_texts)
        except Exception:
            lda_model, vec, X = train_lda_on_texts(cleaned_texts, n_components=max(2, max_topics), max_iter=10)

    # Topic outputs
    topic_distributions = lda_model.transform(X)
    top_topic_ids = np.argsort(topic_distributions.mean(axis=0))[::-1][:max_topics]
    feature_names = vec.get_feature_names_out()

    topics_out = []
    for tid in top_topic_ids:
        comp = lda_model.components_[tid]
        top_idx = comp.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_idx]
        score = float(comp.sum())
        topics_out.append({"topic_id": int(tid), "label": f"Topic {tid}", "keywords": keywords, "score": score})

    # Topic time series - per-document membership in this batch
    timestamps = [int(time.time()) + i for i in range(len(texts))]
    topic_time_series = []
    for tid in top_topic_ids:
        scores = []
        ts_vals = []
        for i, doc_dist in enumerate(topic_distributions):
            scores.append(float(doc_dist[tid]))
            ts_vals.append(timestamps[i])
        topic_time_series.append({"topic_id": int(tid), "timestamps": ts_vals, "scores": scores})

    # Summaries: extractive + optional abstractive
    summaries = []
    abstractive = get_abstractive_pipeline() if use_abstractive else None
    for t in texts:
        ext = extractive_reduce(t, ratio=0.25)
        if abstractive:
            try:
                trimmed = ext[:4000]
                out = abstractive(trimmed, max_length=120, min_length=20, do_sample=False)
                abstr = out[0].get("summary_text", "").strip() if out else ext
            except Exception as e:
                logger.warning(f"Abstractive summarization failed for a doc: {e}")
                abstr = ext
            summaries.append({"extractive": ext, "abstractive": abstr})
        else:
            summaries.append({"extractive": ext})

    # Simple recommendations
    recommendations = []
    avg_topic_weights = topic_distributions.mean(axis=0)
    prominent = np.argsort(avg_topic_weights)[-3:][::-1]
    for tid in prominent:
        top_kw = ", ".join([feature_names[i] for i in lda_model.components_[tid].argsort()[-5:][::-1]])
        recommendations.append({
            "title": f"Investigate topic {tid}",
            "description": f"Topic {tid} is prominent in this data. Top keywords: {top_kw}",
            "confidence": float(min(0.99, avg_topic_weights[tid])),
            "actions": [{"label": "Open topic", "url": f"/topic/{tid}"}]
        })

    return {
        "summaries": summaries,
        "topics": topics_out,
        "topic_time_series": topic_time_series,
        "recommendations": recommendations
    }

# ---------------------------
# Streamlit frontend (as a function)
# ---------------------------
def run_streamlit_app(backend_url: str):
    """
    When run via streamlit, this function will be executed.
    Streamlit will run the file and call this function when mode == frontend.
    """
    try:
        import streamlit as st
        import httpx
        import plotly.express as px
        from streamlit_lottie import st_lottie
    except Exception as e:
        print("Streamlit or dependencies are missing. Install streamlit and run again.", file=sys.stderr)
        raise

    st.set_page_config(page_title="Dynamic Text Analysis (Combined)", layout="wide")
    st.title("Dynamic Text Analysis — Combined App")

    ANALYZE_ENDPOINT = f"{backend_url.rstrip('/')}/analyze"
    HEALTH_ENDPOINT = f"{backend_url.rstrip('/')}/health"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Input")
        input_mode = st.radio("Input mode", ["Paste text", "Upload files", "URLs"])
        texts = []
        if input_mode == "Paste text":
            raw = st.text_area("Paste text here. Use '---' to separate documents.", height=200)
            if raw:
                texts = [t.strip() for t in raw.split("---") if t.strip()]
        elif input_mode == "Upload files":
            uploaded = st.file_uploader("Upload one or more files", accept_multiple_files=True)
            if uploaded:
                for f in uploaded:
                    raw = f.read().decode("utf-8", errors="ignore")
                    texts.append(raw)
        else:
            urls_raw = st.text_area("One URL per line", height=150)
            if urls_raw:
                urls = [u.strip() for u in urls_raw.splitlines() if u.strip()]
                with st.spinner("Fetching URLs..."):
                    for u in urls:
                        try:
                            r = httpx.get(u, timeout=10.0)
                            if r.status_code == 200:
                                texts.append(r.text[:20000])
                            else:
                                texts.append(f"[Failed to fetch {u}]")
                        except Exception as e:
                            texts.append(f"[Failed to fetch {u}: {e}]")

        st.markdown(f"Documents: **{len(texts)}**")
        st.header("Options")
        max_topics = st.slider("Max topics", 3, 20, 8)
        use_abstractive = st.checkbox("Use abstractive summarization (T5)", value=False)
        analyze_btn = st.button("Analyze")

        # optional animation
        try:
            r = httpx.get("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json", timeout=5.0)
            if r.status_code == 200:
                st_lottie(r.json(), height=180)
        except Exception:
            pass

    with col2:
        st.header("Results")
        try:
            rr = httpx.get(HEALTH_ENDPOINT, timeout=3.0)
            if rr.status_code == 200:
                st.success("Backend reachable")
        except Exception:
            st.warning("Backend not reachable. Start backend or set correct backend URL.")

        results_area = st.empty()

    if analyze_btn:
        if not texts:
            st.warning("No input documents found.")
        else:
            payload = {"texts": texts, "mode": "batch", "options": {"max_topics": max_topics, "use_abstractive": use_abstractive}}
            with st.spinner("Analyzing..."):
                try:
                    resp = httpx.post(ANALYZE_ENDPOINT, json=payload, timeout=120.0)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    result = None

            if result:
                with results_area.container():
                    st.subheader("Summaries")
                    for i, s in enumerate(result.get("summaries", [])[:10]):
                        st.markdown(f"**Doc {i+1}**")
                        if isinstance(s, dict):
                            st.write("Extractive:")
                            st.write(s.get("extractive", ""))
                            if "abstractive" in s:
                                st.write("Abstractive:")
                                st.write(s.get("abstractive", ""))
                        else:
                            st.write(s)

                    st.subheader("Topics")
                    topics = result.get("topics", [])
                    if topics:
                        df = pd.DataFrame([{"topic_id": t["topic_id"], "label": t["label"], "score": t["score"], "keywords": ", ".join(t["keywords"])} for t in topics])
                        st.dataframe(df)
                        try:
                            fig = px.scatter(df, x="topic_id", y="score", size="score", color="label", hover_data=["keywords"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    else:
                        st.info("No topics returned.")

                    st.subheader("Recommendations")
                    recs = result.get("recommendations", [])
                    if recs:
                        for r in recs:
                            st.markdown(f"**{r.get('title')}**")
                            st.write(r.get("description"))
                            st.caption(f"Confidence: {r.get('confidence')}")
                    else:
                        st.info("No recommendations.")

# ---------------------------
# Runner utilities
# ---------------------------
def run_backend_uvicorn(host: str, port: int, log_level: str = "info"):
    """Run FastAPI backend using uvicorn programmatically (blocking)."""
    try:
        import uvicorn
    except Exception as e:
        print("uvicorn is required to run the backend. Install uvicorn and try again.", file=sys.stderr)
        raise
    config = uvicorn.Config("app_combined:backend_app", host=host, port=port, log_level=log_level, reload=False)
    server = uvicorn.Server(config)
    server.run()  # blocking

def start_backend_in_thread(host: str, port: int):
    """Start backend in a separate daemon thread."""
    import uvicorn
    def target():
        config = uvicorn.Config("app_combined:backend_app", host=host, port=port, log_level="info", reload=False)
        server = uvicorn.Server(config)
        # this call blocks until server exits
        server.run()
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t

def launch_streamlit_subprocess(script_path: str, extra_args: List[str]):
    """Launch streamlit run <script_path> -- <extra_args> as subprocess."""
    cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--", "--mode", "frontend"] + extra_args
    # Spawn and inherit STDIO (so user sees Streamlit logs)
    proc = subprocess.Popen(cmd)
    return proc

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Combined backend + frontend runner")
    parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both", help="Which parts to run")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host (used for backend and both)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (used for backend and both)")
    parser.add_argument("--backend-url", default=None, help="Backend URL to use when running frontend-only")
    args, unknown = parser.parse_known_args()

    script_path = os.path.abspath(__file__)

    if args.mode == "backend":
        # Run backend blocking
        print(f"Starting backend at http://{args.host}:{args.port}")
        # uvicorn requires import path; to let uvicorn reference this module name we expect file name app_combined.py
        # If run under different module name, try programmatic server
        run_backend_uvicorn(args.host, args.port)

    elif args.mode == "frontend":
        backend_url = args.backend_url or f"http://{args.host}:{args.port}"
        # We expect Streamlit to call this file with --mode frontend (handled below)
        # Launch streamlit run against this file; streamlit will execute the file and call run_streamlit_app when mode==frontend
        print(f"Launching Streamlit frontend connecting to backend at {backend_url}")
        # We pass backend URL via environment so the streamlit subprocess can pick it up
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--", "--mode", "frontend"]
        subprocess.run(cmd, env=env)
    else:  # both
        backend_url = f"http://{args.host}:{args.port}"
        print(f"Starting backend in background at {backend_url} and launching frontend...")
        # Start backend in thread
        t = start_backend_in_thread(args.host, args.port)
        # wait a bit for backend to start
        time.sleep(1.5)
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--", "--mode", "frontend"]
        proc = subprocess.Popen(cmd, env=env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Interrupted. Shutting down.")
        finally:
            # When streamlit exits, the whole process will stop; daemon thread will exit
            pass

# ---------------------------
# Dispatch: when streamlit executes file it will pass --mode frontend
# ---------------------------
if __name__ == "__main__":
    # If Streamlit runs this file, we expect an argument --mode frontend passed after "--"
    # Example of Streamlit invocation: streamlit run app_combined.py -- --mode frontend
    # So parse sys.argv directly for that case and call the streamlit app function instead of starting uvicorn.
    if "--mode" in sys.argv:
        # find the mode specified directly in sys.argv (Streamlit passes args after --)
        try:
            m_idx = sys.argv.index("--mode")
            mode_val = sys.argv[m_idx + 1]
        except Exception:
            mode_val = None
        if mode_val == "frontend":
            # run streamlit UI (the environment variable BACKEND_URL should be set by launcher)
            backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
            run_streamlit_app(backend_url)
            sys.exit(0)
    # Otherwise, normal CLI runner
    main()