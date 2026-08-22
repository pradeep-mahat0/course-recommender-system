# 🎓 CourseIQ — Personalized Learning Recommender

> A full-stack machine learning application that solves the **"too many courses, too little time"** problem by recommending the most relevant online courses to each learner — powered by 9 different recommendation algorithms, a clean modular Python codebase, and an interactive Streamlit interface.

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Overview](#-solution-overview)
3. [Live Demo — How It Looks](#-live-demo--how-it-looks)
4. [9 Recommendation Algorithms Explained](#-9-recommendation-algorithms-explained)
5. [Project Architecture](#-project-architecture)
6. [Tech Stack](#-tech-stack)
7. [Dataset](#-dataset)
8. [Getting Started](#-getting-started)
9. [Step-by-Step Usage Guide](#-step-by-step-usage-guide)
10. [Configuration & Hyperparameters](#-configuration--hyperparameters)
11. [Running Tests](#-running-tests)
12. [Run with AI Coding Tools](#-run-with-ai-coding-tools-claude-code--codex--antigravity)
13. [Future Enhancements](#-future-enhancements)
14. [Author](#-author)

---

## 🔍 Problem Statement

The global e-learning market has exploded — platforms like Coursera, edX, and Udemy now host **tens of thousands of courses**. While this is great for learners, it creates a paradox of choice:

> **"If everything is available, how do I know what to learn next?"**

A typical learner faces these challenges:
- 🔎 **Discovery problem** — it's hard to find courses relevant to their background and goals
- ⏱️ **Time scarcity** — no one wants to spend hours browsing through irrelevant options
- 🎯 **Personalisation gap** — generic "most popular" lists ignore individual preferences
- 🆕 **Cold-start problem** — new users have no history, yet still need good recommendations

This project — **CourseIQ** — solves these problems by learning from a user's past course interactions and producing ranked, personalised course suggestions in seconds.

---

## 💡 Solution Overview

The system takes a user's list of previously taken courses and runs them through one of 9 recommendation models to generate a ranked list of courses the user is most likely to enjoy next.

**What makes this project different:**

| Approach | What most apps do | What this app does |
|----------|------------------|--------------------|
| Models | Usually 1 algorithm | 9 algorithms, user can compare |
| Training | Pre-trained offline | Train models **live in the browser** |
| Tuning | Fixed hyperparameters | Interactive sliders for every model |
| Architecture | Monolith scripts | Modular, layered, testable Python package |
| Cold-start | Often broken | Handled via user profile construction |

---

## 🖥️ Live Demo — How It Looks

```
┌─────────────────────────────────────────────────────────┐
│  🎓 CourseIQ — Personalized Learning Recommender        │
│                                                         │
│  Select courses you have audited or completed:          │
│  ┌────────────────────────────────────────────────┐    │
│  │ ☑  ML for Everyone          | Machine Learning │    │
│  │ ☑  Python for Data Science  | Programming      │    │
│  │ ☐  Deep Learning Basics     | AI               │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  Your selected courses:                                 │
│  • ML for Everyone                                      │
│  • Python for Data Science                              │
└─────────────────────────────────────────────────────────┘

Sidebar:
  1. Select model:         [Course Similarity ▼]
  2. Tune hyperparameters: Top courses [10] | Threshold [50%]
  3. Training:             [Train Model]
  4. Prediction:           [Recommend New Courses]
```

---

## 🧠 9 Recommendation Algorithms Explained

### 1. 📐 Course Similarity
**Type:** Content-Based Filtering

**How it works:**
- Uses a pre-computed **course×course similarity matrix** (cosine similarity on Bag-of-Words representations)
- For each course the user has taken, finds all unseen courses above a similarity threshold
- Recommends the most similar courses ranked by similarity score

**Best for:** Users who want more of the same topics they already enjoy.

**Hyperparameters:**
- `Top N courses` — how many recommendations to show
- `Similarity Threshold %` — minimum similarity score to include a course

---

### 2. 👤 User Profile
**Type:** Content-Based Filtering

**How it works:**
- Builds a **genre-weighted user profile vector**: for each course the user rated, multiplies the course's genre features by the rating given
- Computes a dot product between this profile vector and each unseen course's genre vector
- Courses with the highest alignment with the user's profile are recommended

**Best for:** Capturing a user's overall subject-area interests.

**Hyperparameters:**
- `Top N courses`
- `Profile Similarity Threshold` — minimum dot-product score

---

### 3. 🔵 Clustering (K-Means)
**Type:** Collaborative Filtering via Clustering

**How it works:**
1. All users are represented as **genre-profile vectors**
2. K-Means groups users into clusters of similar learning preferences
3. For a new user, their cluster is predicted, and courses most popular within that cluster are recommended

**Best for:** Finding community-driven recommendations ("learners like you also took...").

**Hyperparameters:**
- `Number of Clusters (K)`
- `Top N courses`

---

### 4. 📉 Clustering with PCA
**Type:** Collaborative Filtering via Dimensionality Reduction + Clustering

**How it works:**
- Same as Clustering, but first applies **Principal Component Analysis (PCA)** to reduce the high-dimensional user profile into a compact representation
- PCA removes noise and correlations, leading to cleaner, more separated clusters

**Best for:** Larger datasets where genre features are highly correlated.

**Hyperparameters:**
- `Number of PCA Components`
- `Number of Clusters (K)`
- `Top N courses`

---

### 5. 🤝 KNN (K-Nearest Neighbours)
**Type:** Collaborative Filtering

**How it works — User-based:**
- Finds the K users most similar to the current user (by their rating vectors)
- Recommends courses rated highly by those similar neighbours

**How it works — Item-based:**
- For each course the user has rated, finds the K most similar courses (by their user-rating vectors)
- Scores unseen courses based on weighted neighbourhood ratings

**Best for:** When you have a dense rating matrix with many users.

**Hyperparameters:**
- `k (neighbors)` — neighbourhood size
- `CF Type` — user-based or item-based
- `Top N courses`

---

### 6. 🧩 NMF (Non-negative Matrix Factorisation)
**Type:** Collaborative Filtering (Matrix Factorisation)

**How it works:**
- Factorises the user×item rating matrix into two non-negative latent factor matrices (user factors × item factors)
- The dot product of a user's factor vector and an item's factor vector gives the predicted rating
- Uses the [Surprise](https://surpriselib.com/) library for efficient implementation

**Best for:** Sparse rating matrices, discovering latent topic-based preferences.

**Hyperparameters:**
- `Latent factors (k)`
- `Epochs`
- `Regularisation (reg_pu, reg_qi)`
- `Top N courses`

---

### 7. 🧬 Neural Network (Embedding Model)
**Type:** Deep Collaborative Filtering

**How it works:**
- A custom Keras model (`RecommenderNet`) learns a **low-dimensional embedding vector** for every user and every course
- The predicted rating = `relu(dot(user_emb, item_emb) + user_bias + item_bias)`
- After training, these embeddings are saved and reused for fast prediction (dot product similarity)

**Architecture:**
```
Input: [user_id, course_id]
    │
    ├─ User Embedding (size=16) ──┐
    ├─ User Bias (size=1)         │
    ├─ Item Embedding (size=16) ──┤→ dot product → + biases → relu → rating
    └─ Item Bias (size=1) ────────┘
```

**Best for:** Capturing non-linear user-item interactions at scale.

**Hyperparameters:**
- `Embedding Size`
- `Epochs`
- `Batch Size`
- `Top N courses`

---

### 8. 📈 Regression with Embedding Features
**Type:** Supervised Learning on Neural Network Embeddings

**How it works:**
- Uses the embeddings trained by the Neural Network model (must train NN first)
- Feature for each (user, course) pair = element-wise sum of their embedding vectors
- Trains a **Linear Regression** model to predict the exact rating score

**Best for:** When you need a continuous score prediction from learned representations.

**Hyperparameters:**
- `Top N courses`

---

### 9. 🏷️ Classification with Embedding Features
**Type:** Supervised Learning on Neural Network Embeddings

**How it works:**
- Same embedding features as Regression (must train NN first)
- Binarises ratings: ≥ 3.0 = "like" (1), < 3.0 = "dislike" (0)
- Trains a **Random Forest Classifier** to predict probability of liking a course
- Recommends courses with the highest predicted "like" probability

**Best for:** When you care about engagement (will the user interact?) more than exact ratings.

**Hyperparameters:**
- `Number of Trees (n_estimators)`
- `Max Tree Depth`
- `Top N courses`

---

## 🏗️ Project Architecture

The codebase is organised into clearly separated layers:

```
personalised-course-recommendation/
│
├── app/                            ← UI layer (Streamlit only)
│   ├── main.py                     ← Entry point
│   └── components/
│       ├── course_selector.py      ← AgGrid interactive course picker
│       └── sidebar.py              ← Model selection + hyperparameter controls
│
├── recommender/                    ← Business / ML logic layer (no Streamlit)
│   ├── config.py                   ← All file paths & MODEL_NAMES constants
│   ├── data_loader.py              ← All CSV I/O in one place
│   ├── engine.py                   ← Dispatcher: train() / predict()
│   └── models/
│       ├── base.py                 ← Abstract BaseRecommender interface
│       ├── content_based.py        ← Models 1 & 2
│       ├── clustering.py           ← Models 3 & 4
│       ├── knn.py                  ← Model 5
│       ├── nmf.py                  ← Model 6
│       ├── neural_net.py           ← Model 7 (RecommenderNet class)
│       └── embedding_models.py     ← Models 8 & 9
│
├── data/                           ← CSV data files
│   ├── ratings.csv                 ← User-course interaction data
│   ├── course_processed.csv        ← Course metadata (ID, title, description)
│   ├── courses_bows.csv            ← Bag-of-Words representation
│   ├── sim.csv                     ← Pre-computed course similarity matrix
│   ├── course_genres_df.csv        ← Course genre/topic feature matrix
│   ├── user_profile_df.csv         ← Pre-built user genre profiles
│   ├── cluster_df.csv              ← KMeans cluster assignments
│   ├── cluster_pca_df.csv          ← PCA+KMeans cluster assignments
│   ├── test_users_df.csv           ← User-course interaction for cluster scoring
│   ├── user_embeddings.csv         ← Learned user embedding vectors (post-NN training)
│   └── course_embeddings.csv       ← Learned course embedding vectors (post-NN training)
│
├── models/                         ← Serialised model artefacts
│   ├── KMeans_model.joblib
│   ├── KMeans_with_pca.joblib
│   ├── pca_model.joblib
│   ├── scaler.joblib
│   ├── knn_sklearn.joblib
│   ├── nmf_model.joblib
│   ├── nn_recommender.keras
│   ├── regression_model.joblib
│   └── classification_model.joblib
│
├── notebooks/                      ← Exploratory & lab Jupyter notebooks
│   ├── lab_jupyter_eda.ipynb
│   ├── lab_jupyter_cf_knn.ipynb
│   ├── lab_jupyter_cf_nmf.ipynb
│   ├── lab_jupyter_cf_ann.ipynb
│   └── ...
│
├── tests/
│   └── test_loaders.py             ← Smoke tests for loaders & engine
│
├── requirements.txt
└── README.md
```

**Separation of Concerns:**

| Layer | Responsibility | Cannot import from |
|-------|---------------|-------------------|
| `app/` | Streamlit UI, user interaction | — |
| `recommender/engine.py` | Routing only, no ML logic | `app/` |
| `recommender/models/` | All ML algorithms | `app/` |
| `recommender/data_loader.py` | All file I/O | `app/`, `engine.py`, `models/` |
| `recommender/config.py` | Constants & paths only | everything |

---

## 🛠️ Tech Stack

| Category | Library | Purpose |
|----------|---------|---------|
| **UI** | Streamlit 1.48 | Interactive web app |
| **UI Component** | streamlit-aggrid | Filterable, selectable data grid |
| **Data** | Pandas 2.3, NumPy 2.3 | Data manipulation |
| **ML — Classical** | scikit-learn 1.5 | KMeans, KNN, PCA, Random Forest, Linear Regression, StandardScaler |
| **ML — Matrix Factorisation** | scikit-surprise 1.1 | NMF collaborative filtering |
| **ML — Deep Learning** | TensorFlow / Keras 2.18 | Neural network embedding model |
| **Sparse Matrices** | SciPy 1.14 | CSR matrix for KNN |
| **Model Serialisation** | joblib 1.4 | Save/load trained models |
| **Testing** | pytest 8.3 | Smoke tests |

---

## 📊 Dataset

The system uses the **IBM Course Ratings dataset** consisting of:

| File | Description | Size |
|------|-------------|------|
| `ratings.csv` | User–course interactions with ratings (1–3 scale) | ~50k+ rows |
| `course_processed.csv` | Course ID, title, and description | ~3.5k courses |
| `course_genres_df.csv` | Binary genre/topic matrix per course (14 genre columns) | — |
| `user_profile_df.csv` | Aggregated genre-rating profile for each user | — |
| `sim.csv` | Pre-computed cosine similarity matrix (courses × courses) | — |
| `courses_bows.csv` | Bag-of-Words token counts per course | — |

**Data Schema — `ratings.csv`:**
```
user   | item        | rating
-------|-------------|-------
1000   | ML0101EN    | 3.0
1000   | PY0101EN    | 3.0
1001   | DS0101EN    | 2.0
```

**Rating Scale:** 2.0 = audited, 3.0 = completed

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9 – 3.11** (TensorFlow does not yet support Python 3.12+)
- `pip` package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/pradeep-mahat0/course-recommender-system.git
cd course-recommender-system
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **TensorFlow note:** If TensorFlow fails to install, the app still works for all models **except** Neural Network, Regression with Embeddings, and Classification with Embeddings. Install TF separately if needed:
> ```bash
> pip install tensorflow==2.18.0
> ```

### 4. Run the Application

```bash
streamlit run app/main.py
```

The app will open automatically in your browser at **http://localhost:8501**

---

## 📖 Step-by-Step Usage Guide

### Step 1 — Select Your Courses

When the app loads, you'll see an **interactive course grid**. Use the checkboxes to select courses you have already taken or are familiar with.

- You can **filter** courses by clicking the column headers
- You can **search** using the sidebar filters in the grid
- Select **at least 1 course** before running predictions

### Step 2 — Choose a Recommendation Model

In the left sidebar, under **"1. Select recommendation model"**, pick the algorithm you want to use:

| Model | Needs Training? | Best When... |
|-------|----------------|--------------|
| Course Similarity | ❌ No | You want topically similar courses |
| User Profile | ❌ No | You want genre-matched courses |
| Clustering | ✅ Yes | You want community-based picks |
| Clustering with PCA | ✅ Yes | Same but with dimensionality reduction |
| KNN | ✅ Yes | You want neighbour-based CF |
| NMF | ✅ Yes | You want latent-factor CF |
| Neural Network | ✅ Yes (slow) | You want deep-learning based picks |
| Regression w/ Embeddings | ✅ NN first | You want rating-score predictions |
| Classification w/ Embeddings | ✅ NN first | You want like/dislike probability |

### Step 3 — Tune Hyperparameters

Under **"2. Tune hyperparameters"**, adjust the sliders to control model behaviour. Each model shows only its relevant controls.

### Step 4 — Train the Model (if required)

Under **"3. Training"**, click the **"Train Model"** button.

- **Course Similarity** and **User Profile** skip training (no button needed)
- **Neural Network** training takes 1–5 minutes depending on your machine
- **Regression / Classification** require the Neural Network to be trained first

### Step 5 — Get Recommendations

Under **"4. Prediction"**, click **"Recommend New Courses"**.

The app will:
1. Register your selected courses as a new user session
2. Run the chosen model
3. Display a ranked table of recommended courses with scores

---

## ⚙️ Configuration & Hyperparameters

All model file paths and dataset paths are defined in one place:

```python
# recommender/config.py
DATA_DIR   = "data/"
MODELS_DIR = "models/"
```

If you move the project or rename directories, **only `config.py` needs to change** — nothing else in the codebase has hardcoded paths.

### Key Hyperparameter Reference

| Model | Parameter | Default | Effect |
|-------|-----------|---------|--------|
| Course Similarity | `sim_threshold` | 50% | Higher = fewer but more similar results |
| User Profile | `profile_sim_threshold` | 15 | Higher = only strong genre matches |
| Clustering | `cluster_no` | 20 | More clusters = more granular grouping |
| KNN | `k` | 40 | More neighbours = smoother but slower |
| NMF | `n_factors` | 50 | More factors = richer but slower model |
| Neural Net | `embedding_size` | 16 | Larger = more expressive, more data needed |
| Classification | `n_estimators` | 100 | More trees = better accuracy, slower |

---

## 🧪 Running Tests

The project includes smoke tests that verify all data loaders and the engine registry work correctly:

```bash
# From the project root
python -m pytest tests/ -v
```

**Expected output:**
```
tests/test_loaders.py::test_load_ratings_has_expected_columns  PASSED
tests/test_loaders.py::test_load_courses_has_expected_columns  PASSED
tests/test_loaders.py::test_load_course_genres_has_course_id   PASSED
tests/test_loaders.py::test_load_bow_has_doc_columns           PASSED
tests/test_loaders.py::test_get_doc_dicts_round_trip           PASSED
tests/test_loaders.py::test_all_model_names_resolve            PASSED (or SKIPPED if TF unavailable)
tests/test_loaders.py::test_unknown_model_raises_value_error   PASSED
```

---

## 🤖 Run with AI Coding Tools (Claude Code / Codex / Antigravity)

This project is structured to work seamlessly with modern AI coding assistants. Below are **ready-to-use prompts** you can paste directly into each tool to get started, explore the codebase, or extend the project.

---

### 🟣 Claude Code

> Claude Code is Anthropic's agentic coding tool. Open it in your terminal inside the project folder.

**Start the application:**
```
Read the project structure, install dependencies from requirements.txt, then start the Streamlit app by running: streamlit run app/main.py
```

**Understand the codebase:**
```
Read through the recommender/ package — config.py, data_loader.py, engine.py, and all files inside models/. Explain how the 9 recommendation models are connected and how a train/predict call flows from app/main.py through to the individual model class.
```

**Add a new model:**
```
I want to add a 10th recommendation model called "SVD" using scikit-surprise. Follow the same pattern as recommender/models/nmf.py — create recommender/models/svd.py with an SVDRecommender class, register it in recommender/engine.py, and add its hyperparameter controls to app/components/sidebar.py.
```

**Debug an issue:**
```
The Clustering model is throwing a FileNotFoundError when I click "Recommend New Courses" without clicking "Train Model" first. Read recommender/models/clustering.py and recommender/config.py and add a clear error message that tells the user to train the model first before predicting.
```

---

### 🟢 OpenAI Codex (via GitHub Copilot / API)

> Use these prompts in GitHub Copilot Chat, the Codex API, or any Codex-powered editor.

**Start the application:**
```
This is a Streamlit-based course recommender system. The entry point is app/main.py. Install dependencies with `pip install -r requirements.txt` and run the app with `streamlit run app/main.py`. Walk me through the setup.
```

**Understand the codebase:**
```
Look at the files in the recommender/ package. The engine.py file dispatches train() and predict() calls. Each model in recommender/models/ inherits from BaseRecommender. Explain the full data flow from when a user clicks "Recommend New Courses" in the UI to when results are displayed.
```

**Add a new feature:**
```
Add a new sidebar section in app/components/sidebar.py that lets the user choose between "Fast" (top 5 courses) and "Detailed" (top 20 courses) recommendation mode, and pass this choice through to the params dict.
```

**Run tests:**
```
Run the existing smoke tests with `python -m pytest tests/ -v` and explain what each test checks. Then write two additional tests in tests/test_loaders.py — one that checks load_user_profile() returns a DataFrame with a 'user' column, and one that checks add_new_ratings() returns an integer.
```

---

### 🔵 Antigravity (Google Deepmind)

> Antigravity is an AI coding assistant with agentic file editing, terminal access, and browser tools. Use these prompts in the Antigravity IDE chat.

**Start the application:**
```
@[path/to/personalised course recommendation] Install the dependencies from requirements.txt, then run the Streamlit app using `streamlit run app/main.py` and confirm it loads correctly.
```

**Understand and explore the codebase:**
```
@[path/to/personalised course recommendation] Go through the entire codebase — start from recommender/config.py, then data_loader.py, engine.py, and each file in recommender/models/. Summarise what each file does, how the 9 models differ from each other, and where a developer would go to add a new model.
```

**Refactor or improve:**
```
@[path/to/personalised course recommendation] Review all files in recommender/models/ and app/components/. Check for any code duplication, missing type hints, or inconsistent naming conventions, and fix them while keeping all existing functionality working.
```

**Extend with a new model:**
```
@[path/to/personalised course recommendation] Add a new BM25-based content similarity model. Create recommender/models/bm25.py with a BM25Recommender class following the same BaseRecommender pattern as content_based.py. Register it in engine.py and add sidebar controls in app/components/sidebar.py. Use the courses_bows.csv data already in the data/ folder.
```

**Run and verify tests:**
```
@[path/to/personalised course recommendation] Run `python -m pytest tests/ -v` and show me the results. If any tests fail, read the relevant source files and fix the issue.
```

---

> 💡 **Tip:** Replace `path/to/personalised course recommendation` with the actual path on your machine when using Antigravity's `@[...]` file reference syntax.

---

## 🔮 Future Enhancements

- [ ] **Hybrid Model** — combine content-based + collaborative scores with a weighted ensemble
- [ ] **Explainable Recommendations** — show *why* a course was recommended (genre match, neighbour overlap, etc.)
- [ ] **User Authentication** — persist user history across sessions with a database
- [ ] **A/B Testing Framework** — compare model performance with click-through tracking
- [ ] **REST API Layer** — expose `/recommend` endpoint so mobile apps can consume results
- [ ] **Real-time Model Evaluation** — display RMSE / precision@K metrics after each training run
- [ ] **Course Metadata Enrichment** — pull live course data via Coursera / edX APIs
- [ ] **Multi-language Support** — extend to non-English course catalogs

---

## 📁 File Quick Reference

| File | What it does |
|------|-------------|
| `app/main.py` | Streamlit page, entry point — run this |
| `app/components/sidebar.py` | All sidebar widgets |
| `app/components/course_selector.py` | AgGrid course table |
| `recommender/config.py` | All constants and paths |
| `recommender/data_loader.py` | All CSV loading functions |
| `recommender/engine.py` | `train()` / `predict()` dispatcher |
| `recommender/models/base.py` | `BaseRecommender` abstract class |
| `recommender/models/content_based.py` | Course Similarity + User Profile |
| `recommender/models/clustering.py` | KMeans + PCA Clustering |
| `recommender/models/knn.py` | KNN Collaborative Filtering |
| `recommender/models/nmf.py` | NMF via Surprise |
| `recommender/models/neural_net.py` | RecommenderNet (Keras) |
| `recommender/models/embedding_models.py` | Regression + Classification |
| `tests/test_loaders.py` | Smoke tests |

---

## 👨‍💻 Author

**Pradeep Mahato**

- GitHub: [@pradeep-mahat0](https://github.com/pradeep-mahat0)
- Repository: [course-recommender-system](https://github.com/pradeep-mahat0/course-recommender-system)

---

> If you found this project useful, please consider giving it a ⭐ on GitHub!
