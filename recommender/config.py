"""Central configuration: all paths and constants in one place."""
import os

# Project root — parent of this file's directory (recommender/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure artifact directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical model names — used as keys throughout app and engine
# ---------------------------------------------------------------------------
MODEL_NAMES = (
    "Course Similarity",
    "User Profile",
    "Clustering",
    "Clustering with PCA",
    "KNN",
    "NMF",
    "Neural Network",
    "Regression with Embedding Features",
    "Classification with Embedding Features",
)

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
RATINGS_PATH            = os.path.join(DATA_DIR, "ratings.csv")
SIM_PATH                = os.path.join(DATA_DIR, "sim.csv")
COURSES_PATH            = os.path.join(DATA_DIR, "course_processed.csv")
BOW_PATH                = os.path.join(DATA_DIR, "courses_bows.csv")
COURSE_GENRES_PATH      = os.path.join(DATA_DIR, "course_genres_df.csv")
USER_PROFILE_PATH       = os.path.join(DATA_DIR, "user_profile_df.csv")
CLUSTER_DF_PATH         = os.path.join(DATA_DIR, "cluster_df.csv")
CLUSTER_PCA_PATH        = os.path.join(DATA_DIR, "cluster_pca_df.csv")
TEST_USERS_PATH         = os.path.join(DATA_DIR, "test_users_df.csv")
USER_EMBEDDINGS_PATH    = os.path.join(DATA_DIR, "user_embeddings.csv")
COURSE_EMBEDDINGS_PATH  = os.path.join(DATA_DIR, "course_embeddings.csv")

# ---------------------------------------------------------------------------
# Saved model artifact paths
# ---------------------------------------------------------------------------
SCALER_PATH              = os.path.join(MODELS_DIR, "scaler.joblib")
KMEANS_PATH              = os.path.join(MODELS_DIR, "KMeans_model.joblib")
KMEANS_PCA_PATH          = os.path.join(MODELS_DIR, "KMeans_with_pca.joblib")
PCA_PATH                 = os.path.join(MODELS_DIR, "pca_model.joblib")
KNN_MODEL_PATH           = os.path.join(MODELS_DIR, "knn_sklearn.joblib")
NMF_MODEL_PATH           = os.path.join(MODELS_DIR, "nmf_model.joblib")
NN_MODEL_PATH            = os.path.join(MODELS_DIR, "nn_recommender.keras")
REGRESSION_MODEL_PATH    = os.path.join(MODELS_DIR, "regression_model.joblib")
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "classification_model.joblib")
