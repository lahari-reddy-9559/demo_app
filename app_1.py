import os
import time
import json
from typing import List, Dict, Any
import httpx
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
from dotenv import load_dotenv

load_dotenv()

# CONFIG
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ANALYZE_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/analyze"
HEALTH_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/health"

st.set_page_config(page_title="Dynamic Text Analysis", layout="wide", initial_sidebar_state="expanded")

# helper to load example lottie animation from web
def load_lottie_from_url(url: str):
    try:
        r = httpx.get(url, timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

# small adapter to call backend
def call_analyze_api(texts: List[str], mode: str = "batch", options: Dict[str, Any] = None) -> Dict[str, Any]:
    payload = {"texts": texts, "mode": mode, "options": options or {}}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(ANALYZE_ENDPOINT, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        st.error(f"Failed to call backend analyze API: {e}")
        return {}

def check_health() -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(HEALTH_ENDPOINT)
            if r.status_code == 200:
                return r.json()
    except Exception:
        return {}

# UI layout
st.title("Dynamic Text Analysis — Interactive App")
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.header("Input")
    input_mode = st.radio("Input mode", ["Paste / Short text", "Upload files (txt, csv)", "Provide URLs"], index=0)
    texts = []

    if input_mode == "Paste / Short text":
        text_input = st.text_area("Paste text here (multiple paragraphs allowed). Use '---' to separate documents.", height=200)
        if text_input:
            # split by delimiter into documents
            docs = [t.strip() for t in text_input.split("---") if t.strip()]
            texts.extend(docs)

    elif input_mode == "Upload files (txt, csv)":
        uploaded = st.file_uploader("Upload one or more files", accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                try:
                    raw = f.read().decode("utf-8")
                except Exception:
                    raw = f.read().decode("latin-1")
                if f.name.lower().endswith(".csv"):
                    df = pd.read_csv(pd.io.common.StringIO(raw))
                    # If there's a 'text' column take it, otherwise join all string columns
                    if "text" in df.columns:
                        texts.extend(df["text"].dropna().astype(str).tolist())
                    else:
                        # try find first text-like column
                        text_cols = [c for c in df.columns if df[c].dtype == object]
                        if text_cols:
                            texts.extend(df[text_cols[0]].dropna().astype(str).tolist())
                else:
                    texts.append(raw)

    else:  # URLs
        url_list_raw = st.text_area("One URL per line", height=150)
        if url_list_raw:
            urls = [u.strip() for u in url_list_raw.splitlines() if u.strip()]
            # Simple fetcher - backend could also accept URLs directly
            with st.spinner("Fetching URLs..."):
                for u in urls:
                    try:
                        r = httpx.get(u, timeout=15.0)
                        if r.status_code == 200:
                            texts.append(r.text[:20000])  # cap
                        else:
                            texts.append(f"[Failed to fetch {u}: status {r.status_code}]")
                    except Exception as e:
                        texts.append(f"[Failed to fetch {u}: {e}]")

    st.markdown(f"Detected documents: **{len(texts)}**")

    st.header("Options")
    mode = st.selectbox("Processing Mode", ["batch", "stream"], index=0)
    max_topics = st.slider("Max topics to display", min_value=3, max_value=25, value=8)
    enable_recommendations = st.checkbox("Show recommendations", value=True)
    analyze_btn = st.button("Analyze")

    # Lottie animation for the input column
    lottie_anim = load_lottie_from_url("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_anim:
        st_lottie(lottie_anim, height=200)

with col2:
    st.header("Results")
    health = check_health()
    if health:
        st.success(f"Backend healthy — {health.get('model','unknown')}")

    summary_area = st.empty()
    topics_area = st.empty()
    timeseries_area = st.empty()
    recs_area = st.empty()

# Main analyze flow
if analyze_btn:
    if not texts:
        st.warning("No input texts found. Paste text, upload files, or provide URLs.")
    else:
        with st.spinner("Analyzing... This may take a while depending on backend models..."):
            result = call_analyze_api(texts, mode=mode, options={"max_topics": max_topics})
        if not result:
            st.error("No result returned from backend.")
        else:
            # Summaries
            summaries = result.get("summaries", [])
            topics = result.get("topics", [])
            timeseries = result.get("topic_time_series", [])
            recommendations = result.get("recommendations", [])

            # Show combined summary
            with summary_area.container():
                st.subheader("Summaries")
                if summaries:
                    for i, s in enumerate(summaries[:5]):
                        st.markdown(f"**Doc {i+1}:** {s}")
                else:
                    st.info("No summaries returned.")

            # Topics table + bubble chart
            with topics_area.container():
                st.subheader("Topics and Keywords")
                if topics:
                    df_topics = pd.DataFrame([{
                        "topic_id": t.get("topic_id"),
                        "label": t.get("label"),
                        "score": t.get("score", 0.0),
                        "keywords": ", ".join(t.get("keywords", [])[:6])
                    } for t in topics])
                    st.dataframe(df_topics.sort_values("score", ascending=False).reset_index(drop=True), height=240)

                    # bubble chart of topics by score
                    fig = px.scatter(df_topics, x="topic_id", y="score", size="score", color="label",
                                     hover_data=["keywords"], title="Topic importance (bubble size = score)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No topics returned.")

            # Topic time series animated
            with timeseries_area.container():
                st.subheader("Topic Evolution")
                if timeseries:
                    # Flatten into a DataFrame for plotting
                    rows = []
                    for t in timeseries:
                        tid = t.get("topic_id")
                        ts = t.get("timestamps", [])
                        scores = t.get("scores", [])
                        for ts_i, sc in zip(ts, scores):
                            rows.append({"topic_id": tid, "timestamp": pd.to_datetime(ts_i), "score": sc})
                    if rows:
                        df_ts = pd.DataFrame(rows)
                        # pivot for stacked area
                        df_plot = df_ts.sort_values("timestamp")
                        fig2 = px.area(df_plot, x="timestamp", y="score", color="topic_id",
                                       title="Topic scores over time", line_group="topic_id")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No time series points returned.")
                else:
                    st.info("No topic time series returned.")

            # Recommendations
            with recs_area.container():
                st.subheader("Recommendations")
                if enable_recommendations and recommendations:
                    for rec in recommendations:
                        st.markdown(f"### {rec.get('title','Recommendation')}")
                        st.write(rec.get("description",""))
                        st.caption(f"Confidence: {rec.get('confidence', 0.0):.2f}")
                        actions = rec.get("actions", [])
                        if actions:
                            cols = st.columns(len(actions))
                            for c, a in zip(cols, actions):
                                if c.button(a.get("label","Action")):
                                    # action could be a URL or an instruction to open a modal; we simply open URL
                                    url = a.get("url")
                                    if url:
                                        st.experimental_set_query_params(action=url)
                                        st.write(f"Open: {url}")
                else:
                    st.info("No recommendations returned or disabled.")

        st.success("Analysis complete.")

# Footer / tips
st.markdown("---")
st.markdown("Tips: Connect BACKEND_URL in your environment to point to your trained model API. Make sure CORS is allowed. For streaming mode, adapt the analyze endpoint to send partial updates and extend this frontend to consume Server-Sent Events or websockets.")
