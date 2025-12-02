# 🎓 Personalized Learning Recommender System

A powerful and interactive **course recommendation system** built with **Streamlit**, combining both traditional machine-learning algorithms and deep-learning models to help learners discover the best courses tailored to their personal preferences.

---

## 📌 **Overview**

The explosion of online education platforms has empowered millions—but it also overwhelms learners with too many choices. This project solves that problem by providing **personalized course recommendations** using:

* Content-based filtering
* Collaborative filtering
* Clustering
* Deep learning models

The system offers an intuitive **Streamlit interface** where users can select courses they’ve taken, choose a model, tune hyperparameters, train real-time, and instantly view recommendations.

---

## 🚀 **Features**

* **Interactive UI**: Built with Streamlit + AgGrid for fast filtering and selection.
* **8 Recommendation Algorithms** .
* **Real-time Model Training**: Train Neural Networks or KNN directly in the browser.
* **Hyperparameter Tuning**: Sliders for k-value, epochs, clusters, etc.
* **Cold-Start Support**: Automatically builds a user profile for new users.
* **Embeddings-Based Models**: Neural networks + regression/classification using learned embeddings.

---

## 🧠 **Models Implemented**

### **1. Content-Based Filtering**

* Cosine similarity over processed course descriptions.

### **2. User Profile Model**

* Builds a weighted user vector based on genre affinity.

### **3. K-Means Clustering**

* Recommends courses popular within the user's cluster.

### **4. PCA + Clustering**

* Dimensionality reduction for improved clustering performance.

### **5. KNN (User-Based / Item-Based)**

* k-Nearest Neighbors collaborative filtering.

### **6. Neural Network Recommender**

* Custom Keras model learning low-dimensional embeddings.

### **7. Regression with Embeddings**

* Linear regression using NN-learned embeddings.

### **8. Classification with Embeddings**

* Random Forest classifier using learned embeddings.

*(Note: NMF model removed as per latest project update.)*

---

## 🛠️ **Installation**

Follow the steps to run the project locally:

### **1. Clone the Repository**

```bash
git clone https://github.com/pradeep-mahat0/course-recommender-system.git
cd course-recommender-system
```

### **2. Create a Virtual Environment (Optional)**

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the Application**

```bash
streamlit run recommender_app.py
```

---

## 📖 **Usage Guide**

1. **Launch the app** — the course dataset loads into an interactive grid.
2. **Select courses** you’ve previously taken or liked.
3. **Choose a model** from the sidebar.
4. **Tune hyperparameters** using sliders.
5. **Train the model** (only for training-based models like Neural Network).
6. Click **Recommend New Courses** to see personalized suggestions.

---

## 📂 **Project Structure**

```
├── recommender_app.py      # Main Streamlit UI
├── backend.py              # Model controller and logic layer
├── nn_model.py             # Keras-based embedding model
├── helper.py               # Utility functions for processing & clustering
├── recommend.py            # Cluster recommendation logic
├── data/                   # CSV datasets
│   ├── ratings.csv
│   ├── course_processed.csv
│   └── ...
└── requirements.txt
```

---

## 🔮 **Future Enhancements**

* **Database Integration** (PostgreSQL/MySQL) for scalable user data.
* **Hybrid Filtering** combining content + collaborative scores.
* **Explainable Recommendations** with SHAP, attention maps, or feature attributions.
* **API Layer** to allow mobile app or external service integration.

---

### 👨‍💻 **Created by *Pradeep Mahato***

If you like this project, don’t forget to ⭐ the repository!
