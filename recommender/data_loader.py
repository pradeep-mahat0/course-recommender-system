"""
Data loading utilities.
All I/O is centralised here — nothing else in the codebase reads CSVs directly.
"""
import pandas as pd

from recommender.config import (
    RATINGS_PATH,
    SIM_PATH,
    COURSES_PATH,
    BOW_PATH,
    COURSE_GENRES_PATH,
    USER_PROFILE_PATH,
    CLUSTER_DF_PATH,
    CLUSTER_PCA_PATH,
    TEST_USERS_PATH,
    USER_EMBEDDINGS_PATH,
    COURSE_EMBEDDINGS_PATH,
)


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(RATINGS_PATH)


def load_course_sims() -> pd.DataFrame:
    return pd.read_csv(SIM_PATH)


def load_courses() -> pd.DataFrame:
    df = pd.read_csv(COURSES_PATH)
    df["TITLE"] = df["TITLE"].str.title()
    return df


def load_bow() -> pd.DataFrame:
    return pd.read_csv(BOW_PATH)


def load_course_genres() -> pd.DataFrame:
    return pd.read_csv(COURSE_GENRES_PATH)


def load_user_profile() -> pd.DataFrame:
    return pd.read_csv(USER_PROFILE_PATH)


def load_cluster_df() -> pd.DataFrame:
    return pd.read_csv(CLUSTER_DF_PATH)


def load_cluster_pca_df() -> pd.DataFrame:
    return pd.read_csv(CLUSTER_PCA_PATH)


def load_test_users() -> pd.DataFrame:
    return pd.read_csv(TEST_USERS_PATH)


def load_user_embeddings() -> pd.DataFrame:
    return pd.read_csv(USER_EMBEDDINGS_PATH)


def load_course_embeddings() -> pd.DataFrame:
    return pd.read_csv(COURSE_EMBEDDINGS_PATH)


def get_doc_dicts() -> tuple[dict, dict]:
    """Return (idx→course_id, course_id→idx) mappings derived from the BOW file."""
    bow_df = load_bow()
    grouped = bow_df.groupby(["doc_index", "doc_id"]).max().reset_index()
    idx_id: dict = grouped[["doc_id"]].to_dict()["doc_id"]
    id_idx: dict = {v: k for k, v in idx_id.items()}
    return idx_id, id_idx


def add_new_ratings(new_course_ids: list) -> int | None:
    """
    Append a synthetic new user (rating=3.0 for all selected courses) to ratings.csv.

    Returns the new user ID, or None if no courses were provided.
    Note: callers in the Streamlit layer must clear st.cache_data after calling this
    so subsequent loads pick up the updated file.
    """
    if not new_course_ids:
        return None

    ratings_df = load_ratings()
    new_id = int(ratings_df["user"].max()) + 1

    new_rows = pd.DataFrame({
        "user":   [new_id] * len(new_course_ids),
        "item":   new_course_ids,
        "rating": [3.0]    * len(new_course_ids),
    })

    updated = pd.concat([ratings_df, new_rows], ignore_index=True)
    updated.to_csv(RATINGS_PATH, index=False)
    return new_id
