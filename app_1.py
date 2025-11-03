#!/usr/bin/env python3
"""
Combined app with deferred backend imports.

Usage:
  - Backend only:
      python app_combined_safe.py --mode backend --host 127.0.0.1 --port 8000
  - Frontend only:
      python app_combined_safe.py --mode frontend --backend-url http://127.0.0.1:8000
  - Both (starts backend in thread then launches Streamlit):
      python app_combined_safe.py --mode both --host 127.0.0.1 --port 8000

This version delays importing FastAPI/uvicorn until backend code is executed.
"""
import argparse
import os
import sys
import threading
import time
import subprocess
import logging
from typing import List, Optional

# Lightweight imports safe for Streamlit-only environments
import numpy as np
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_safe")

# NLTK setup (best-effort)
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
    from nltk.corpus import stopwords
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

# Lazy backend creation function — imports FastAPI only when needed
def create_backend_app():
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except Exception as e:
        raise RuntimeError("FastAPI is not installed in this environment. Install fastapi and try again.") from e

    import numpy as _np
    from nltk.tokenize import word_tokenize as _word_tokenize

    backend = FastAPI(title="Deferred Combined Backend")

    class AnalyzeOptions(BaseModel):
        max_topics: Optional[int] = 8
        use_abstractive: Optional[bool] = True

    class AnalyzeRequest(BaseModel):
        texts: List[str]
        mode: Optional[str] = "batch"
        options: Optional[AnalyzeOptions] = AnalyzeOptions()

    # In-memory state
    STATE = {
        "tfidf_vectorizer": None,
        "lda_model": None,
        "lda_trained_on": None,
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

    @backend.get("/health")
    def health():
        return {"status": "ok", "models_loaded": {k: (v is not None) for k, v in STATE.items()}}

    @backend.post("/analyze")
    def analyze(req: AnalyzeRequest):
        texts = req.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided.")
        options = req.options or AnalyzeOptions()
        max_topics = int(options.max_topics or 8)
        use_abstractive = bool(options.use_abstractive)

        cleaned_texts = [clean_text(t) for t in texts]

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

        topic_distributions = lda_model.transform(X)
        top_topic_ids = _np.argsort(topic_distributions.mean(axis=0))[::-1][:max_topics]
        feature_names = vec.get_feature_names_out()

        topics_out = []
        for tid in top_topic_ids:
            comp = lda_model.components_[tid]
            top_idx = comp.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_idx]
            score = float(comp.sum())
            topics_out.append({"topic_id": int(tid), "label": f"Topic {tid}", "keywords": keywords, "score": score})

        # Summaries (extractive only here)
        summaries = []
        for t in texts:
            ext = extractive_reduce(t, ratio=0.25)
            summaries.append({"extractive": ext})

        # Recommendations
        recommendations = []
        avg_topic_weights = topic_distributions.mean(axis=0)
        prominent = _np.argsort(avg_topic_weights)[-3:][::-1]
        for tid in prominent:
            top_kw = ", ".join([feature_names[i] for i in lda_model.components_[tid].argsort()[-5:][::-1]])
            recommendations.append({
                "title": f"Investigate topic {tid}",
                "description": f"Topic {tid} is prominent. Top keywords: {top_kw}",
                "confidence": float(min(0.99, avg_topic_weights[tid])),
                "actions": [{"label": "Open topic", "url": f"/topic/{tid}"}]
            })

        return {
            "summaries": summaries,
            "topics": topics_out,
            "topic_time_series": [],
            "recommendations": recommendations
        }

    return backend

# Run backend in thread using uvicorn, but import uvicorn only when starting backend
def start_backend_thread(host: str, port: int):
    try:
        import uvicorn
    except Exception as e:
        raise RuntimeError("uvicorn not installed. Install uvicorn to run backend.") from e

    backend_app = create_backend_app()

    def target():
        config = uvicorn.Config(backend_app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        server.run()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t

# Streamlit UI runner (same as before)
def run_streamlit_app(backend_url: str):
    try:
        import streamlit as st
        import httpx
        import plotly.express as px
        from streamlit_lottie import st_lottie
    except Exception as e:
        print("Streamlit or dependencies missing. Install streamlit and run again.", file=sys.stderr)
        raise

    st.set_page_config(page_title="Dynamic Text Analysis (Safe)", layout="wide")
    st.title("Dynamic Text Analysis — Safe Combined App")

    ANALYZE_ENDPOINT = f"{backend_url.rstrip('/')}/analyze"
    HEALTH_ENDPOINT = f"{backend_url.rstrip('/')}/health"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Input")
        raw = st.text_area("Paste text here. Use '---' to separate docs.", height=200)
        texts = [t.strip() for t in raw.split("---") if t.strip()] if raw else []
        st.markdown(f"Documents: **{len(texts)}**")
        max_topics = st.slider("Max topics", 3, 20, 8)
        analyze_btn = st.button("Analyze")
    with col2:
        st.header("Results")
        results_area = st.empty()

    if analyze_btn:
        if not texts:
            st.warning("No texts provided.")
        else:
            payload = {"texts": texts, "mode": "batch", "options": {"max_topics": max_topics, "use_abstractive": False}}
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
                    for i, s in enumerate(result.get("summaries",[])[:10]):
                        st.markdown(f"**Doc {i+1}**")
                        st.write(s.get("extractive","") if isinstance(s,dict) else s)
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
                    st.subheader("Recommendations")
                    for r in result.get("recommendations", []):
                        st.markdown(f"**{r.get('title')}**")
                        st.write(r.get("description"))
                        st.caption(f"Confidence: {r.get('confidence')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backend","frontend","both"], default="both")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default=None)
    args, _ = parser.parse_known_args()

    if args.mode == "backend":
        # Run backend (blocking)
        print(f"Starting backend at http://{args.host}:{args.port}")
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed. Install uvicorn to run backend.", file=sys.stderr)
            sys.exit(1)
        backend_app = create_backend_app()
        uvicorn.run(backend_app, host=args.host, port=args.port)
    elif args.mode == "frontend":
        backend_url = args.backend_url or f"http://{args.host}:{args.port}"
        os.environ["BACKEND_URL"] = backend_url
        # Streamlit runs this file with --mode frontend; launching via streamlit CLI is expected.
        run_streamlit_app(backend_url)
    else:  # both
        backend_url = f"http://{args.host}:{args.port}"
        try:
            start_backend_thread(args.host, args.port)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        # wait a moment for server to come up (increase if models load slowly)
        time.sleep(1.5)
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), "--", "--mode", "frontend"]
        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    # If streamlit passes --mode frontend, we need to call run_streamlit_app
    if "--mode" in sys.argv:
        try:
            idx = sys.argv.index("--mode")
            mode_val = sys.argv[idx + 1]
        except Exception:
            mode_val = None
        if mode_val == "frontend":
            backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
            run_streamlit_app(backend_url)
            sys.exit(0)
    main()
