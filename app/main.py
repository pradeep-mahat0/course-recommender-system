"""
Personalised Learning Recommender — Streamlit entry point.

Run from the project root:
    streamlit run app/main.py

Bug fixes applied:
  - Bug #8: st.cache_data.clear() is called after writing new ratings to disk,
            so the next load_ratings() call sees the updated file.
"""
import time

import pandas as pd
import streamlit as st

import recommender.engine as engine
from recommender.data_loader import load_courses, add_new_ratings
from app.components.course_selector import render_course_selector
from app.components.sidebar import render_sidebar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personalised Learning Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached data loaders (Streamlit layer) ─────────────────────────────────────
@st.cache_data
def _cached_courses() -> pd.DataFrame:
    return load_courses()


# ── App init ──────────────────────────────────────────────────────────────────
def _init_app() -> pd.DataFrame:
    """Load data, render the course picker, and return the user's selection."""
    placeholder = st.empty()
    with st.spinner("Loading datasets…"):
        course_df = _cached_courses()
    placeholder.success("Datasets loaded successfully.")
    time.sleep(1)
    placeholder.empty()

    selected_df = render_course_selector(course_df)
    st.subheader("Your selected courses:")
    st.table(selected_df)
    return selected_df


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Personalised Learning Recommender")

selected_courses_df         = _init_app()
model_name, params          = render_sidebar()

# ── Training ──────────────────────────────────────────────────────────────────
st.sidebar.subheader("3. Training")
if st.sidebar.button("Train Model"):
    with st.spinner(f"Training {model_name}…"):
        engine.train(model_name, params)
    st.success(f"✅ {model_name} trained successfully!")

# ── Prediction ────────────────────────────────────────────────────────────────
st.sidebar.subheader("4. Prediction")
if st.sidebar.button("Recommend New Courses"):
    if selected_courses_df.empty:
        st.warning("Please select at least one course first.")
    else:
        new_id = add_new_ratings(selected_courses_df["COURSE_ID"].tolist())

        # Bug #8 fix: flush the Streamlit data cache so the mutated
        # ratings.csv is visible to all subsequent load_ratings() calls.
        st.cache_data.clear()

        with st.spinner("Generating recommendations…"):
            time.sleep(0.3)
            res_df = engine.predict(model_name, [new_id], params)

        if res_df.empty:
            st.info(
                "No recommendations found with the current settings. "
                "Try lowering the similarity threshold."
            )
        else:
            st.success("🎯 Recommendations generated!")
            course_df  = _cached_courses()
            display_df = (
                res_df[["COURSE_ID", "SCORE"]]
                .merge(course_df, on="COURSE_ID", how="left")
                .drop(columns=["COURSE_ID"])
                .sort_values("SCORE", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(display_df)
