"""
Smoke tests for data loaders and the engine dispatcher.
Run from the project root:   pytest tests/ -v
"""
import pytest
import pandas as pd

from recommender.data_loader import (
    load_ratings,
    load_courses,
    load_course_genres,
    load_bow,
    get_doc_dicts,
)
from recommender.config import MODEL_NAMES
import recommender.engine as engine


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

def test_load_ratings_has_expected_columns():
    df = load_ratings()
    assert isinstance(df, pd.DataFrame)
    assert {"user", "item", "rating"}.issubset(df.columns), \
        f"Missing columns. Got: {df.columns.tolist()}"
    assert len(df) > 0, "ratings.csv is empty"


def test_load_courses_has_expected_columns():
    df = load_courses()
    assert isinstance(df, pd.DataFrame)
    assert "COURSE_ID" in df.columns
    assert "TITLE" in df.columns


def test_load_course_genres_has_course_id():
    df = load_course_genres()
    assert "COURSE_ID" in df.columns


def test_load_bow_has_doc_columns():
    df = load_bow()
    assert "doc_index" in df.columns
    assert "doc_id" in df.columns


def test_get_doc_dicts_round_trip():
    idx_id, id_idx = get_doc_dicts()
    assert isinstance(idx_id, dict) and len(idx_id) > 0
    assert isinstance(id_idx, dict) and len(id_idx) > 0
    # Every mapping must round-trip correctly
    for idx, cid in list(idx_id.items())[:10]:
        assert id_idx[cid] == idx, f"Round-trip failed for idx={idx}, cid={cid}"


# ---------------------------------------------------------------------------
# Engine registry tests
# ---------------------------------------------------------------------------

def test_all_model_names_resolve():
    """Every name in MODEL_NAMES must map to an instantiable class."""
    tf_models = {MODEL_NAMES[6], MODEL_NAMES[7], MODEL_NAMES[8]}
    for name in MODEL_NAMES:
        if name in tf_models:
            try:
                import tensorflow  # noqa: F401
            except ImportError:
                pytest.skip(f"TensorFlow not available — skipping {name}")
        obj = engine.get_model(name)
        assert hasattr(obj, "train"),   f"{name}: missing train()"
        assert hasattr(obj, "predict"), f"{name}: missing predict()"


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model"):
        engine.get_model("FakeModel")
