"""
Dispatcher: routes train() and predict() calls to the correct model class.

This is the single entry-point that app/main.py talks to.
No model logic lives here — it only instantiates and delegates.

Imports for TensorFlow-dependent models are deferred (lazy) so this module
can be imported even when TF is unavailable (e.g., during unit tests that
don't exercise the Neural Network model).
"""
import importlib

import pandas as pd

from recommender.config import MODEL_NAMES
from recommender.data_loader import add_new_ratings  # re-exported for convenience
from recommender.models.content_based import (
    CourseSimilarityRecommender,
    UserProfileRecommender,
)
from recommender.models.clustering import (
    ClusteringRecommender,
    ClusteringPCARecommender,
)
from recommender.models.knn import KNNRecommender
from recommender.models.nmf import NMFRecommender

# ---------------------------------------------------------------------------
# Registry: model name → (module_path, class_name)
# TF-dependent models use lazy loading so this module stays importable even
# when TensorFlow has environment issues (e.g., protobuf version conflicts).
# ---------------------------------------------------------------------------
_LAZY_REGISTRY: dict[str, tuple[str, str]] = {
    MODEL_NAMES[6]: ("recommender.models.neural_net",     "NeuralNetRecommender"),
    MODEL_NAMES[7]: ("recommender.models.embedding_models", "RegressionRecommender"),
    MODEL_NAMES[8]: ("recommender.models.embedding_models", "ClassificationRecommender"),
}

_EAGER_REGISTRY: dict[str, type] = {
    MODEL_NAMES[0]: CourseSimilarityRecommender,
    MODEL_NAMES[1]: UserProfileRecommender,
    MODEL_NAMES[2]: ClusteringRecommender,
    MODEL_NAMES[3]: ClusteringPCARecommender,
    MODEL_NAMES[4]: KNNRecommender,
    MODEL_NAMES[5]: NMFRecommender,
}


def get_model(model_name: str):
    """Return a fresh instance of the requested model."""
    if model_name in _EAGER_REGISTRY:
        return _EAGER_REGISTRY[model_name]()

    if model_name in _LAZY_REGISTRY:
        module_path, class_name = _LAZY_REGISTRY[model_name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls()

    all_names = list(_EAGER_REGISTRY) + list(_LAZY_REGISTRY)
    raise ValueError(f"Unknown model: {model_name!r}. Available: {all_names}")


def train(model_name: str, params: dict) -> None:
    """Fit the named model with the given hyperparameters."""
    get_model(model_name).train(params)


def predict(model_name: str, user_ids: list, params: dict) -> pd.DataFrame:
    """
    Generate recommendations for the given user IDs.

    Returns
    -------
    pd.DataFrame  columns: USER, COURSE_ID, SCORE  (sorted by SCORE desc)
    """
    result  = get_model(model_name).predict(user_ids, params)
    top_n   = params.get("top_courses")
    if top_n:
        return result.head(int(top_n))
    return result
