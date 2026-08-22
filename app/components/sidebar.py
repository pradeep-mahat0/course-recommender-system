"""
Sidebar component: model selection and per-model hyperparameter controls.

Extracting this from main.py gives a single place to add or adjust
hyperparameter widgets without touching the page layout code.
"""
import streamlit as st
from recommender.config import MODEL_NAMES


def render_sidebar() -> tuple[str, dict]:
    """
    Render all sidebar controls and collect hyperparameters.

    Returns
    -------
    model_name : str
    params     : dict   hyperparameter values ready to pass to engine.train/predict
    """
    st.sidebar.subheader("1. Select recommendation model")
    model_name = st.sidebar.selectbox("Select model:", MODEL_NAMES)

    params: dict = {}
    st.sidebar.subheader("2. Tune hyperparameters")

    # ── Course Similarity ────────────────────────────────────────────────────
    if model_name == MODEL_NAMES[0]:
        params["top_courses"]    = st.sidebar.slider("Top courses", 0, 20, 10, 1)
        params["sim_threshold"]  = st.sidebar.slider(
            "Course Similarity Threshold %", 0, 100, 50, 10
        )

    # ── User Profile ─────────────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[1]:
        params["top_courses"]           = st.sidebar.slider("Top courses", 0, 20, 10, 1)
        params["profile_sim_threshold"] = st.sidebar.slider(
            "Profile Similarity Threshold", 0, 50, 15, 5
        )

    # ── Clustering ───────────────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[2]:
        params["cluster_no"]  = st.sidebar.slider("Number of Clusters", 2, 30, 20, 1)
        params["top_courses"] = st.sidebar.slider("Top courses", 0, 20, 10, 1)

    # ── Clustering with PCA ──────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[3]:
        params["n_components"] = st.sidebar.slider("PCA Components", 1, 14, 9, 1)
        params["cluster_no"]   = st.sidebar.slider("Number of Clusters", 2, 30, 20, 1)
        params["top_courses"]  = st.sidebar.slider("Top courses", 0, 20, 10, 1)

    # ── KNN ──────────────────────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[4]:
        params["top_courses"] = st.sidebar.slider("Top courses", 1, 30, 10, 1)
        params["k"]           = st.sidebar.slider("k (neighbors)", 5, 100, 40, 5)
        params["user_based"]  = (
            st.sidebar.radio("CF Type", ["user-based", "item-based"]) == "user-based"
        )

    # ── NMF ──────────────────────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[5]:
        params["top_courses"] = st.sidebar.slider("Top courses", 1, 30, 10, 1)
        params["n_factors"]   = st.sidebar.slider("Latent factors", 10, 200, 50, 10)
        params["n_epochs"]    = st.sidebar.slider("Epochs", 5, 100, 20, 5)
        params["reg_pu"]      = st.sidebar.number_input("Reg pu", 0.0, 1.0, 0.06, 0.01)
        params["reg_qi"]      = st.sidebar.number_input("Reg qi", 0.0, 1.0, 0.06, 0.01)

    # ── Neural Network ───────────────────────────────────────────────────────
    elif model_name == MODEL_NAMES[6]:
        st.sidebar.info("Trains user/item embeddings. Prediction reuses saved embeddings.")
        params["embedding_size"] = st.sidebar.slider("Embedding Size", 8, 32, 16, 4)
        params["epochs"]         = st.sidebar.slider("Epochs", 1, 20, 5, 1)
        params["batch_size"]     = st.sidebar.slider("Batch Size", 16, 128, 64, 16)
        params["top_courses"]    = st.sidebar.slider("Top courses", 1, 30, 10, 1)

    # ── Regression with Embedding Features ───────────────────────────────────
    elif model_name == MODEL_NAMES[7]:
        st.sidebar.warning("Train 'Neural Network' first to generate embeddings.")
        params["top_courses"] = st.sidebar.slider("Top courses", 1, 30, 10, 1)

    # ── Classification with Embedding Features ───────────────────────────────
    elif model_name == MODEL_NAMES[8]:
        st.sidebar.warning("Train 'Neural Network' first to generate embeddings.")
        params["top_courses"]  = st.sidebar.slider("Top courses", 1, 30, 10, 1)
        params["n_estimators"] = st.sidebar.slider("Number of Trees", 50, 200, 100, 10)
        params["max_depth"]    = st.sidebar.slider("Max Tree Depth", 5, 20, 10, 1)

    return model_name, params
