#!/usr/bin/env python3
"""
Combined Streamlit frontend + optional FastAPI backend (safe/deferred imports).

Save as app_combined_safe.py.

Usage examples:

1) Run frontend only (assumes backend already running at BACKEND_URL or default http://127.0.0.1:8000):
   BACKEND_URL=http://127.0.0.1:8000 python app_combined_safe.py --mode frontend

2) Run backend only (blocking):
   python app_combined_safe.py --mode backend --host 127.0.0.1 --port 8000

3) Run both locally (starts backend in thread then launches Streamlit in same environment):
   python app_combined_safe.py --mode both --host 127.0.0.1 --port 8000

Notes:
- This file defers heavy imports (fastapi, uvicorn, transformers, nltk resources) so the Streamlit frontend can run
  in environments that don't have backend packages installed.
- For abstractive summarization or advanced NLP you'll need to install additional packages (transformers, torch, nltk).
"""

import argparse
import os
import sys
import time
import threading
import subprocess
import logging
from typing import List, Optional

# Lightweight imports (safe)
import math
import re
import heapq
import string
import numpy as np

# Optional: attempt to make some NLP helpers available without failing if nltk isn't installed.
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
    _NLTK_AVAILABLE = True
except Exception:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    word_tokenize = None
    _NLTK_AVAILABLE = False

# Try to ensure some NLTK resources if available
if _NLTK_AVAILABLE:
    try:
        for pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
            try:
                nltk.data.find(pkg)
            except Exception:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer() if _NLTK_AVAILABLE else None
_stop_words = set()
if _NLTK_AVAILABLE:
    try:
        _stop_words = set(stopwords.words("english"))
    except Exception:
        _stop_words = set()

def safe_word_tokenize(text: str):
    if _NLTK_AVAILABLE and word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    # fallback
    return re.findall(r"\w+", text.lower())

def get_stopwords_set():
    return _stop_words if _stop_words else set()

STOP_WORDS = get_stopwords_set()

def safe_lemmatize(word: str):
    if lemmatizer is not None:
        try:
            return lemmatizer.lemmatize(word)
        except Exception:
            return word
    return word

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [safe_lemmatize(w) for w in words]
    return " ".join(words)

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 8) -> str:
    """Simple extractive reduction: score sentences by token frequency and pick top ones."""
    if not text or not isinstance(text, str):
        return ""
    _SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return text.strip()

    def word_tokens(s: str):
        toks = re.findall(r"\w+", s.lower())
        return [t for t in toks if t not in STOP_WORDS]

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

# In-memory lightweight state for optional models
STATE = {
    "tfidf_vectorizer": None,
    "lda_model": None,
    "lda_trained_on": None,
    "abstractive_pipeline": None,  # loaded lazily if transformers available
}

# Scikit-learn imports are safe to assume for basic TF-IDF / LDA; if not installed, backend mode will error clearly.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.decomposition import LatentDirichletAllocation  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    LatentDirichletAllocation = None
    _SKLEARN_AVAILABLE = False

def train_lda_on_texts(texts: List[str], n_components: int = 5, max_iter: int = 10):
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required to train LDA. Install scikit-learn.")
    vec = TfidfVectorizer(max_features=5000)
    cleaned = [clean_text(t) for t in texts]
    X = vec.fit_transform(cleaned)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, random_state=42)
    lda.fit(X)
    STATE["tfidf_vectorizer"] = vec
    STATE["lda_model"] = lda
    STATE["lda_trained_on"] = cleaned
    return lda, vec, X

# Deferred creation of FastAPI app to avoid import errors in frontend-only envs.
def create_backend_app():
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
        from pydantic import BaseModel  # type: ignore
    except Exception as e:
        raise RuntimeError("FastAPI/pydantic not installed in this environment. Install fastapi and pydantic to run backend.") from e

    try:
        import numpy as _np  # type: ignore
    except Exception:
        raise RuntimeError("numpy is required for backend. Install numpy.")

    app = FastAPI(title="Combined Dynamic Text Analysis (deferred backend)")

    class AnalyzeOptions(BaseModel):
        max_topics: Optional[int] = 8
        use_abstractive: Optional[bool] = False

    class AnalyzeRequest(BaseModel):
        texts: List[str]
        mode: Optional[str] = "batch"
        options: Optional[AnalyzeOptions] = AnalyzeOptions()

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": {k: (v is not None) for k, v in STATE.items()}}

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest):
        texts = req.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided.")
        opts = req.options or AnalyzeOptions()
        max_topics = int(opts.max_topics or 8)
        use_abstractive = bool(opts.use_abstractive)

        cleaned_texts = [clean_text(t) for t in texts]
        tokenized_texts = [safe_word_tokenize(ct) for ct in cleaned_texts]

        # Train or reuse LDA
        if STATE.get("lda_model") is None or STATE.get("lda_trained_on") is None:
            if not _SKLEARN_AVAILABLE:
                raise HTTPException(status_code=500, detail="scikit-learn not available in backend environment.")
            lda_model, vec, X = train_lda_on_texts(cleaned_texts, n_components=max(2, max_topics), max_iter=10)
        else:
            lda_model = STATE["lda_model"]
            vec = STATE["tfidf_vectorizer"]
            try:
                X = vec.transform(cleaned_texts)
            except Exception:
                lda_model, vec, X = train_lda_on_texts(cleaned_texts, n_components=max(2, max_topics), max_iter=10)

        topic_distributions = lda_model.transform(X)
        top_topic_ids = list(np.argsort(topic_distributions.mean(axis=0))[::-1][:max_topics])
        feature_names = vec.get_feature_names_out()

        topics_out = []
        for tid in top_topic_ids:
            comp = lda_model.components_[tid]
            top_idx = comp.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_idx]
            score = float(comp.sum())
            topics_out.append({"topic_id": int(tid), "label": f"Topic {tid}", "keywords": keywords, "score": score})

        # Topic time series (simple per-doc membership returned with timestamps)
        timestamps = [int(time.time()) + i for i in range(len(texts))]
        topic_time_series = []
        for tid in top_topic_ids:
            scores = []
            ts_vals = []
            for i, doc_dist in enumerate(topic_distributions):
                scores.append(float(doc_dist[tid]))
                ts_vals.append(timestamps[i])
            topic_time_series.append({"topic_id": int(tid), "timestamps": ts_vals, "scores": scores})

        # Summaries
        summaries = []
        # Optionally load abstractive pipeline if requested and transformers available
        abstractive_pipeline = None
        if use_abstractive:
            try:
                from transformers import pipeline as hf_pipeline  # type: ignore
                # load lazily and cache in STATE
                if STATE.get("abstractive_pipeline") is None:
                    try:
                        STATE["abstractive_pipeline"] = hf_pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)
                    except Exception:
                        STATE["abstractive_pipeline"] = None
                abstractive_pipeline = STATE["abstractive_pipeline"]
            except Exception:
                abstractive_pipeline = None

        for t in texts:
            ext = extractive_reduce(t, ratio=0.25)
            if abstractive_pipeline is not None:
                try:
                    trimmed = ext[:4000]
                    out = abstractive_pipeline(trimmed, max_length=120, min_length=20, do_sample=False)
                    abstr = out[0].get("summary_text", "").strip() if out else ext
                except Exception:
                    abstr = ext
                summaries.append({"extractive": ext, "abstractive": abstr})
            else:
                summaries.append({"extractive": ext})

        # Recommendations (simple rule-based)
        recommendations = []
        avg_topic_weights = topic_distributions.mean(axis=0)
        prominent = list(np.argsort(avg_topic_weights)[-3:][::-1])
        for tid in prominent:
            top_kw = ", ".join([feature_names[i] for i in lda_model.components_[tid].argsort()[-5:][::-1]])
            recommendations.append({
                "title": f"Investigate topic {tid}",
                "description": f"Topic {tid} shows prominence. Top keywords: {top_kw}",
                "confidence": float(min(0.99, avg_topic_weights[tid])),
                "actions": [{"label": "Open topic", "url": f"/topic/{tid}"}]
            })

        return {
            "summaries": summaries,
            "topics": topics_out,
            "topic_time_series": topic_time_series,
            "recommendations": recommendations,
        }

    return app

# Threaded backend starter (imports uvicorn lazily)
def start_backend_thread(host: str, port: int):
    try:
        import uvicorn  # type: ignore
    except Exception:
        raise RuntimeError("uvicorn not installed. Install uvicorn to run backend.")

    backend_app = create_backend_app()

    def target():
        config = uvicorn.Config(backend_app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        server.run()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t

# Streamlit UI - executed when running frontend mode (or when Streamlit runs this file)
def run_streamlit_app(backend_url: str):
    try:
        import streamlit as st  # type: ignore
        import httpx  # type: ignore
        import plotly.express as px  # type: ignore
        # streamlit_lottie is optional
        try:
            from streamlit_lottie import st_lottie  # type: ignore
        except Exception:
            st_lottie = None
    except Exception as e:
        print("Streamlit or its dependencies are not installed in this environment.", file=sys.stderr)
        raise

    ANALYZE_ENDPOINT = f"{backend_url.rstrip('/')}/analyze"
    HEALTH_ENDPOINT = f"{backend_url.rstrip('/')}/health"

    st.set_page_config(page_title="Dynamic Text Analysis (Safe)", layout="wide")
    st.title("Dynamic Text Analysis — Combined (Safe)")

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.header("Input")
        input_mode = st.radio("Input mode", ["Paste text", "Upload files", "URLs"])
        texts = []
        if input_mode == "Paste text":
            raw = st.text_area("Paste text here. Use '---' to separate documents.", height=220)
            if raw:
                texts = [t.strip() for t in raw.split("---") if t.strip()]
        elif input_mode == "Upload files":
            uploaded = st.file_uploader("Upload one or more text files", accept_multiple_files=True)
            if uploaded:
                for f in uploaded:
                    try:
                        raw = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        try:
                            raw = f.read().decode("latin-1", errors="ignore")
                        except Exception:
                            raw = ""
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
                                texts.append(f"[Failed to fetch {u}: status={r.status_code}]")
                        except Exception as e:
                            texts.append(f"[Failed to fetch {u}: {e}]")

        st.markdown(f"Detected documents: **{len(texts)}**")
        st.header("Options")
        max_topics = st.slider("Max topics to display", 3, 20, 8)
        use_abstractive = st.checkbox("Use abstractive summarization (requires transformers)", value=False)
        analyze_btn = st.button("Analyze")

        # Lottie animation (optional)
        if st_lottie is not None:
            try:
                r = httpx.get("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json", timeout=5.0)
                if r.status_code == 200:
                    st_lottie(r.json(), height=160)
            except Exception:
                pass

    with col2:
        st.header("Results")
        try:
            rr = httpx.get(HEALTH_ENDPOINT, timeout=3.0)
            if rr.status_code == 200:
                st.success("Backend reachable")
            else:
                st.warning("Backend responded with non-200 status.")
        except Exception:
            st.warning("Backend not reachable. Start backend or set BACKEND_URL env var.")

        results_area = st.empty()

    if analyze_btn:
        if not texts:
            st.warning("No texts provided.")
        else:
            payload = {"texts": texts, "mode": "batch", "options": {"max_topics": max_topics, "use_abstractive": use_abstractive}}
            with st.spinner("Analyzing..."):
                try:
                    resp = httpx.post(ANALYZE_ENDPOINT, json=payload, timeout=120.0)
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"Analysis request failed: {e}")
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
                        df = None
                        try:
                            df = __import__("pandas").DataFrame([{
                                "topic_id": t["topic_id"],
                                "label": t["label"],
                                "score": t.get("score", 0.0),
                                "keywords": ", ".join(t.get("keywords", []))
                            } for t in topics])
                            st.dataframe(df)
                            try:
                                fig = px.scatter(df, x="topic_id", y="score", size="score", color="label", hover_data=["keywords"])
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                        except Exception:
                            # fallback simple listing
                            for t in topics:
                                st.markdown(f"- {t.get('label')} (id={t.get('topic_id')}), top: {', '.join(t.get('keywords', [])[:8])}")
                    else:
                        st.info("No topics returned.")

                    st.subheader("Recommendations")
                    recs = result.get("recommendations", [])
                    if recs:
                        for r in recs:
                            st.markdown(f"**{r.get('title')}**")
                            st.write(r.get("description", ""))
                            st.caption(f"Confidence: {r.get('confidence', 0.0):.2f}")
                    else:
                        st.info("No recommendations returned.")
            else:
                st.error("No result from backend.")

def main():
    parser = argparse.ArgumentParser(description="Combined Streamlit + optional FastAPI app (safe imports)")
    parser.add_argument("--mode", choices=["backend", "frontend", "both"], default="both", help="Which parts to run")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host")
    parser.add_argument("--port", type=int, default=8000, help="Backend port")
    parser.add_argument("--backend-url", default=None, help="If running frontend-only, specify backend URL")
    args, unknown = parser.parse_known_args()

    script_path = os.path.abspath(__file__)

    if args.mode == "backend":
        # Run backend using uvicorn (blocking)
        try:
            import uvicorn  # type: ignore
        except Exception:
            print("uvicorn is required to run the backend. Install uvicorn and try again.", file=sys.stderr)
            sys.exit(1)
        backend_app = create_backend_app()
        uvicorn.run(backend_app, host=args.host, port=args.port)

    elif args.mode == "frontend":
        backend_url = args.backend_url or os.environ.get("BACKEND_URL", f"http://{args.host}:{args.port}")
        # Run streamlit UI in current process (useful when streamlit run invokes file with --mode frontend)
        run_streamlit_app(backend_url)

    else:  # both
        backend_url = f"http://{args.host}:{args.port}"
        # start backend in thread
        try:
            start_backend_thread(args.host, args.port)
        except Exception as e:
            print(f"Failed to start backend thread: {e}", file=sys.stderr)
            sys.exit(1)
        # give backend a moment to begin
        time.sleep(1.5)
        # Launch Streamlit as subprocess so it uses its own process + web UI
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--", "--mode", "frontend"]
        proc = subprocess.Popen(cmd, env=env)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    # When Streamlit runs this file with "streamlit run app_combined_safe.py -- --mode frontend",
    # Streamlit will place "--mode frontend" in sys.argv. Detect that and call run_streamlit_app directly.
    if "--mode" in sys.argv:
        try:
            idx = sys.argv.index("--mode")
            mode_val = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        except Exception:
            mode_val = None
        if mode_val == "frontend":
            backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
            run_streamlit_app(backend_url)
            sys.exit(0)
    main()
