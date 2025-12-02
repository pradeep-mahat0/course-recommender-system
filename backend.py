import pandas as pd
import numpy as np
import helper as helper
import joblib
import recommend as recommend
import streamlit as st

# --- NEW IMPORTS ---
import os
from surprise import Dataset, Reader, KNNBasic, NMF

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import nn_model
import tensorflow as tf
from tensorflow import keras

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_course_genre():
    return pd.read_csv("course_genres_df.csv")

def load_user_profile():
    return pd.read_csv("user_profile_df.csv")

def load_cluster_df():
    return pd.read_csv("cluster_df.csv")

def load_test_users():
    return pd.read_csv("test_users_df.csv")

def load_cluster_pca():
    return pd.read_csv("cluster_pca_df.csv")

# --- embedding loaders (new) ---
def load_user_embeddings():
    return pd.read_csv("user_embeddings.csv")

def load_course_embeddings():
    return pd.read_csv("course_embeddings.csv")


def add_new_ratings(new_courses):

    # """this function take input the list of new_courses id and create a new id for this user ,rating 3 to all its selected courses add to the rating table and update it and return new_user id  """

    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1#max id +1

        users = [new_id] * len(new_courses)#all have same id 
        ratings = [3.0] * len(new_courses)# default 3 rating for all 3 courses 

        # fill the res_dict with the columns
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings

        #dict to dataframe
        new_df = pd.DataFrame(res_dict)

        #rating is added to rating df
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


#..........................................
#..........................................
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- HELPER FUNCTIONS FOR NN-BASED MODELS ---

def prepare_data_for_nn(rating_df):
    """Encodes user and item IDs and splits data for training."""
    # Mapping user and item ids to indices
    user_ids = rating_df["user"].unique().tolist()
    user_id2idx = {x: i for i, x in enumerate(user_ids)}
    
    course_ids = rating_df["item"].unique().tolist()
    course_id2idx = {x: i for i, x in enumerate(course_ids)}

    rating_df["user_idx"] = rating_df["user"].map(user_id2idx)
    rating_df["item_idx"] = rating_df["item"].map(course_id2idx)

    num_users = len(user_id2idx)
    num_items = len(course_id2idx)

    # Splitting data
    x = rating_df[["user_idx", "item_idx"]].values
    y = rating_df["rating"].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test, num_users, num_items, user_id2idx, course_id2idx


def prepare_data_for_embedding_models():
    """Prepares the dataset for regression/classification using embeddings."""
    try:
        ratings_df = load_ratings()
        user_emb = load_user_embeddings()
        item_emb = load_course_embeddings()
    except FileNotFoundError:
        print("Embedding files not found. Please train the Neural Network model first.")
        return None, None
        
    # Merge embeddings with ratings
    merged_df = pd.merge(ratings_df, user_emb, how='left', on='user').fillna(0)
    merged_df = pd.merge(merged_df, item_emb, how='left', on='item').fillna(0)
    
    u_features = [f"UFeature{i}" for i in range(16)]
    c_features = [f"CFeature{i}" for i in range(16)]
    
    # Element-wise addition of features
    X = merged_df[u_features] + merged_df[c_features].values
    y = merged_df['rating']
    
    return X, y

def _build_urm():
    ratings = load_ratings()
    users = np.sort(ratings['user'].unique())
    items = np.sort(ratings['item'].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    rows = ratings['user'].map(user2idx).to_numpy()
    cols = ratings['item'].map(item2idx).to_numpy()
    data = ratings['rating'].astype(np.float32).to_numpy()

    M = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return M, user2idx, item2idx, users, items

def _unseen_items_for_user(user_id, items_list):
    ratings = load_ratings()
    seen = set(ratings[ratings['user'] == user_id]['item'].unique())
    return sorted(set(items_list) - seen)

def _ratings_reader():
    # auto-infer rating scale from data
    r = load_ratings()
    mn, mx = float(r['rating'].min()), float(r['rating'].max())
    if mn == mx:  # fallback
        mn, mx = 1.0, 5.0
    return Reader(rating_scale=(mn, mx))

def _surprise_trainset():
    r = load_ratings()
    reader = _ratings_reader()
    data = Dataset.load_from_df(r[['user','item','rating']], reader)
    return data.build_full_trainset()


def _course_feature_matrix():
    """Return (course_ids, X) from course_genres_df.csv where X are genre/feature cols."""
    df = load_course_genre()
    # assumes first two columns are COURSE_ID and maybe TITLE or a count column; adjust if needed
    non_feat_cols = ['COURSE_ID','TITLE']
    feat_cols = [c for c in df.columns if c not in non_feat_cols]
    return df['COURSE_ID'].values, df[feat_cols].values, feat_cols

def _user_profile_vec(user_id):
    ratings_df = load_ratings()
    course_genres_df = load_course_genre()
    # existing helper you already use in other branches:
    return helper.create_user_profile(user_id, ratings_df, course_genres_df)

def _make_content_pairs():
    """Build supervised (X, y) from ratings using user_profile × course_features."""
    ratings = load_ratings()
    course_ids, Xc, feat_cols = _course_feature_matrix()
    cid_to_row = {cid:i for i, cid in enumerate(course_ids)}
    X, y = [], []
    for _, row in ratings.iterrows():
        u = int(row['user']); i = int(row['item']); r = float(row['rating'])
        if i not in cid_to_row: 
            continue
        up = _user_profile_vec(u)                     # shape [F]
        ci = Xc[cid_to_row[i]]                       # shape [F]
        # Use elementwise product (strong signal) + concat (richer)
        prod = up * ci
        X.append(np.concatenate([up, ci, prod], axis=0))
        y.append(r)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

def _make_content_pairs_binary(threshold=3.0):
    X, y = _make_content_pairs()
    yb = (y >= threshold).astype(np.int64)
    return X, yb


#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

def _surprise_trainset():
    r = load_ratings()
    reader = Reader(rating_scale=(1, 5))  # adjust scale if different
    data = Dataset.load_from_df(r[['user','item','rating']], reader)
    return data.build_full_trainset()

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
     

    #  """this function take the 4 params index2item_id ,item_id2index ,course id that are enrolled by the user and similarity matrix and gives a dict in which id of the course with their similarity in desc order """


    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]

                # sim_matrix is typically a 2D matrix (num_courses × num_courses) where:

                # sim_matrix[i][j] = similarity between course i and course j.
                sim = sim_matrix[idx1][idx2]

                #If multiple enrolled courses are compared to the same unselected course, we keep the highest similarity score.
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim

    #Sorts dictionary by similarity values in descending order.
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)} 
    return res


def user_profile_recommendation_scores(idx_id_dict, id_idx_dict, enrolled_course_ids,test_user_vector):
    
     # List to store user IDs
    courses = []    # List to store recommended course IDs
    scores = []     # List to store recommendation scores
    res={}

    
    course_genres_df=load_course_genre()
    enrolled_courses = set(idx_id_dict.values())
    unknown_courses = enrolled_courses.difference(enrolled_course_ids)


    # Filter the course_genres_df to include only unknown courses
    unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].values

    # Calculate the recommendation scores using dot product
    recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

    # Append the results into the users, courses, and scores list
    for i in range(0, len(unknown_course_ids)):
        
        # Only keep the courses with high recommendation score
        
        courses.append(unknown_course_ids[i])
        scores.append(recommendation_scores[i])

    res=dict(zip(courses,scores))
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)} 
    return res




# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if 'cluster_no' in params:
        cluster_no=params['cluster_no']
    if 'n_components' in params:
        n_components=params['n_components']

    if model_name == models[2]:
        user_profile_df=load_user_profile()
        cluster_df=helper.preprocess_and_train_kMeans(user_profile_df,cluster_no)
        cluster_df.to_csv("cluster_df.csv",index=False)

    elif model_name ==models[3]:
        user_profile_df=load_user_profile()
        cluster_pca_df=helper.pca_and_train(user_profile_df,n_components,cluster_no)
        cluster_pca_df.to_csv("cluster_pca_df.csv",index=False)

    elif model_name == models[4]:  # "KNN"
        k = params.get('k', 40)
        user_based = params.get('user_based', True)

        URM, user2idx, item2idx, users_list, items_list = _build_urm()
        X = URM if user_based else URM.T
        X = normalize(X.astype(np.float32), axis=1, norm='l2', copy=True)

        nn = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(X)

        joblib.dump({
            'nn': nn,
            'user2idx': user2idx,
            'item2idx': item2idx,
            'users_list': users_list,
            'items_list': items_list,
            'user_based': user_based,
            'k': k,
            'n_users': URM.shape[0],   # save number of users at training time
        }, os.path.join(MODELS_DIR, 'knn_sklearn.joblib'))

    elif model_name == models[5]:  # "NMF"
        n_factors = params.get('n_factors', 50)
        n_epochs = params.get('n_epochs', 20)
        reg_pu = params.get('reg_pu', 0.06)
        reg_qi = params.get('reg_qi', 0.06)

        trainset = _surprise_trainset()
        algo = NMF(n_factors=n_factors, n_epochs=n_epochs,
                   reg_pu=reg_pu, reg_qi=reg_qi, verbose=False)
        algo.fit(trainset)
        joblib.dump(algo, os.path.join(MODELS_DIR, 'nmf_model.joblib'))

    elif model_name == "Neural Network":
        print("Training Neural Network...")
        # 1. Prepare data
        ratings_df = load_ratings()
        x_train, _, y_train, _, num_users, num_items, user_map, item_map = prepare_data_for_nn(ratings_df)
        
        # 2. Define model
        embedding_size = params.get('embedding_size', 16)
        model = nn_model.RecommenderNet(num_users, num_items, embedding_size)
        
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        # 3. Train model
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 64)
        
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        
        # 4. Save model and embeddings
        model.save(os.path.join(MODELS_DIR, 'nn_recommender.keras'))
        
        # Extract and save user embeddings
        user_latent_features = model.get_layer('user_embedding_layer').get_weights()[0]
        user_ids_df = pd.DataFrame(user_map.keys(), columns=['user'])
        user_emb_df = pd.DataFrame(user_latent_features, columns=[f"UFeature{i}" for i in range(embedding_size)])
        user_emb_df = pd.concat([user_ids_df, user_emb_df], axis=1)
        user_emb_df.to_csv("user_embeddings.csv", index=False)
        
        # Extract and save item embeddings
        item_latent_features = model.get_layer('item_embedding_layer').get_weights()[0]
        item_ids_df = pd.DataFrame(item_map.keys(), columns=['item'])
        item_emb_df = pd.DataFrame(item_latent_features, columns=[f"CFeature{i}" for i in range(embedding_size)])
        item_emb_df = pd.concat([item_ids_df, item_emb_df], axis=1)
        item_emb_df.to_csv("course_embeddings.csv", index=False)
        
        print("Neural Network trained and embeddings saved.")

    elif model_name == "Regression with Embedding Features":
        print("Training Regression model with embeddings...")
        X, y = prepare_data_for_embedding_models()
        if X is None: return
        
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, 'regression_model.joblib'))
        print("Regression model trained.")

    elif model_name == "Classification with Embedding Features":
        print("Training Classification model with embeddings...")
        X, y_raw = prepare_data_for_embedding_models()
        if X is None: return
        
        # Binarize ratings: 3 and above is "like" (1), below is "dislike" (0)
        y = (y_raw >= 3.0).astype(int)
        
        # Using RandomForest as it's robust
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, 'classification_model.joblib'))
        print("Classification model trained.")
    
    # ... (Keep your other model training elif blocks here) ...
    # e.g., elif model_name == models[2]: ...





# Prediction
def predict(model_name, user_ids, params):

    # """input ->
    #    output -> df of user_id,course_id,Score"""
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0

    profile_sim_threshold=10
    if "profile_sim_threshold" in params:
        profile_sim_threshold=params["profile_sim_threshold"]


    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Load user's rated courses to filter them out from recommendations
        ratings_df = load_ratings()
        enrolled_course_ids = ratings_df[ratings_df['user'] == user_id]['item'].tolist()
        all_courses_df = load_courses()

        if model_name in ["Regression with Embedding Features", "Classification with Embedding Features"]:
            try:
                user_emb_df = load_user_embeddings()
                item_emb_df = load_course_embeddings()
            except FileNotFoundError:
                st.error("Embeddings not found. Please train the 'Neural Network' model first.")
                return pd.DataFrame()

            # Get the current user's embedding vector
            user_vector = user_emb_df[user_emb_df['user'] == user_id]
            if user_vector.empty:
                # Handle cold-start: use an average user vector
                user_vector = user_emb_df.drop('user', axis=1).mean().values.reshape(1, -1)
            else:
                user_vector = user_vector.drop('user', axis=1).values

            # Candidate courses are all courses not yet rated by the user
            candidate_courses_df = item_emb_df[~item_emb_df['item'].isin(enrolled_course_ids)]
            
            # Prepare feature matrix for prediction
            item_vectors = candidate_courses_df.drop('item', axis=1).values
            X_pred = user_vector + item_vectors
            
            # Load the correct model and predict
            if model_name == "Regression with Embedding Features":
                model = joblib.load(os.path.join(MODELS_DIR, 'regression_model.joblib'))
                pred_scores = model.predict(X_pred)
            else: # Classification
                model = joblib.load(os.path.join(MODELS_DIR, 'classification_model.joblib'))
                # We want the probability of the "like" class (1)
                pred_scores = model.predict_proba(X_pred)[:, 1]

            # Populate results
            users.extend([user_id] * len(candidate_courses_df))
            courses.extend(candidate_courses_df['item'].tolist())
            scores.extend(pred_scores)
        
        elif model_name == "Neural Network":
            # For NN, we use the saved embeddings to calculate similarity scores
            try:
                user_emb_df = load_user_embeddings()
                item_emb_df = load_course_embeddings()
            except FileNotFoundError:
                st.error("Embeddings not found. Please train the 'Neural Network' model first.")
                return pd.DataFrame()

            user_vector = user_emb_df[user_emb_df['user'] == user_id]
            if user_vector.empty:
                user_vector = user_emb_df.drop('user', axis=1).mean().values.reshape(1, -1)
            else:
                user_vector = user_vector.drop('user', axis=1).values

            candidate_courses_df = item_emb_df[~item_emb_df['item'].isin(enrolled_course_ids)]
            item_vectors = candidate_courses_df.drop('item', axis=1).values
            
            # Calculate dot product as similarity score
            pred_scores = np.dot(item_vectors, user_vector.T).flatten()
            
            users.extend([user_id] * len(candidate_courses_df))
            courses.extend(candidate_courses_df['item'].tolist())
            scores.extend(pred_scores)

        # ... (Keep your other model prediction elif blocks here) ...
        # e.g., elif model_name == models[0]: ...

        # Course Similarity model
        elif model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here
        
        elif model_name ==models[1]:
            ratings_df=load_ratings()
            course_genres_df=load_course_genre()
            user_ratings=ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            
            test_user_vector=helper.create_user_profile(user_id,ratings_df,course_genres_df)
            res=user_profile_recommendation_scores(idx_id_dict, id_idx_dict, enrolled_course_ids,test_user_vector)
            for key, score in res.items():
                if score >= profile_sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        elif model_name == models[2]:

            user_profile_df=load_user_profile()
            ratings_df=load_ratings()
            course_genres_df=load_course_genre()
            user_ratings=ratings_df[ratings_df['user'] == user_id]

            enrolled_course_ids = user_ratings['item'].to_list()
            #making of user_profile_vector
            test_user_vector=helper.create_user_profile(user_id,ratings_df,course_genres_df)
            #adding user_id to the vector
            merged_list=[user_id]
            merged_list.extend(test_user_vector)
            merged_series=pd.Series(merged_list,index=user_profile_df.columns)
            # Convert the Series to a DataFrame
            user_profile_df = merged_series.to_frame().T

            #prediction
            scaler=joblib.load('scaler.joblib')
            model=joblib.load('KMeans_model.joblib')
            cluster=helper.predict_cluster(scaler,model,user_profile_df)
            cluster=cluster.item()



            test_users_df=load_test_users()
            cluster_df=load_cluster_df()

            test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

            # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
            courses_cluster = test_users_labelled[['item', 'cluster']]

            # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
            courses_cluster['count'] = [1] * len(courses_cluster)

            # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
            # and resetting the index to make the result more readable
            courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()

            courses_cluster_grouped=courses_cluster_grouped[courses_cluster_grouped['cluster']==cluster].drop(columns=['cluster'])

            res=recommend.cluster(courses_cluster_grouped,enrolled_course_ids)

            for key, score in res.items():
                
                users.append(user_id)
                courses.append(key)
                scores.append(score)


        elif model_name == models[3]:

            user_profile_df=load_user_profile()
            ratings_df=load_ratings()
            course_genres_df=load_course_genre()
            user_ratings=ratings_df[ratings_df['user'] == user_id]

            enrolled_course_ids = user_ratings['item'].to_list()
            #making of user_profile_vector
            test_user_vector=helper.create_user_profile(user_id,ratings_df,course_genres_df)
            #adding user_id to the vector
            merged_list=[user_id]
            merged_list.extend(test_user_vector)
            merged_series=pd.Series(merged_list,index=user_profile_df.columns)
            # Convert the Series to a DataFrame
            user_profile_df = merged_series.to_frame().T

            #prediction
            scaler=joblib.load('scaler.joblib')
            pca_model=joblib.load('pca_model.joblib')
            model=joblib.load('KMeans_with_pca.joblib')
            cluster=helper.predict_pca_cluster(scaler,pca_model,model,user_profile_df)
            cluster=cluster.item()

            test_users_df=load_test_users()
            cluster_df=load_cluster_pca()

            test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

            # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
            courses_cluster = test_users_labelled[['item', 'cluster']]

            # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
            courses_cluster['count'] = [1] * len(courses_cluster)

            # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
            # and resetting the index to make the result more readable
            courses_cluster_grouped = courses_cluster.groupby(['cluster','item']).agg(enrollments=('count','sum')).reset_index()

            courses_cluster_grouped=courses_cluster_grouped[courses_cluster_grouped['cluster']==cluster].drop(columns=['cluster'])

            res=recommend.cluster(courses_cluster_grouped,enrolled_course_ids)

            for key, score in res.items():
                
                users.append(user_id)
                courses.append(key)
                scores.append(score)

        elif model_name == models[4]:  # "KNN"
            model_data = joblib.load(os.path.join(MODELS_DIR, 'knn_sklearn.joblib'))
            nn = model_data['nn']
            user2idx, item2idx = model_data['user2idx'], model_data['item2idx']
            items_list = model_data['items_list']
            user_based, k = model_data['user_based'], model_data['k']
            URM, _, _, _, _ = _build_urm()

            for user_id in user_ids:
                unseen = _unseen_items_for_user(user_id, items_list)
                if user_based:
                    # Build normalized user vector
                    user_vec = np.zeros(len(items_list), dtype=np.float32)
                    ratings_df = load_ratings()
                    for _, r in ratings_df[ratings_df['user'] == user_id].iterrows():
                        if r['item'] in item2idx:
                            user_vec[item2idx[r['item']]] = float(r['rating'])
                    norm = np.linalg.norm(user_vec)
                    if norm > 0:
                        user_vec /= norm
                    distances, indices = nn.kneighbors(user_vec.reshape(1, -1), n_neighbors=min(k+1, nn._fit_X.shape[0]))
                    sims = 1.0 - distances.ravel()
                    neighbor_idxs = indices.ravel()
                    for iid in unseen:
                        if iid not in item2idx: continue
                        neigh_ratings = URM[neighbor_idxs, item2idx[iid]].toarray().ravel()
                        mask = neigh_ratings != 0
                        if mask.any():
                            est = np.dot(sims[mask], neigh_ratings[mask]) / sims[mask].sum()
                            users.append(user_id); courses.append(iid); scores.append(est)
                else:
                    # item-based CF
                    ratings_df = load_ratings()
                    user_ratings = ratings_df[ratings_df['user'] == user_id]
                    score_dict, weight_sum = {}, {}

                    n_users_trained = model_data['n_users']  # number of users at training time

                    for _, r in user_ratings.iterrows():
                        iid, r_val = r['item'], r['rating']
                        if iid not in item2idx:
                            continue

                        # get this item's column (all user ratings for this item)
                        item_vec = URM[:, item2idx[iid]].toarray().ravel()

                        # adjust to match training dimension
                        if len(item_vec) < n_users_trained:
                            diff = n_users_trained - len(item_vec)
                            item_vec = np.concatenate([item_vec, np.zeros(diff, dtype=np.float32)])
                        elif len(item_vec) > n_users_trained:
                            item_vec = item_vec[:n_users_trained]

                        # normalize
                        norm = np.linalg.norm(item_vec)
                        if norm == 0:
                            continue
                        item_norm = (item_vec / norm).reshape(1, -1)

                        # find neighbors
                        distances, indices = nn.kneighbors(item_norm, n_neighbors=min(k+1, nn._fit_X.shape[0]))
                        sims = 1.0 - distances.ravel()

                        # accumulate neighbor scores
                        for idx, sim in zip(indices.ravel(), sims):
                            candidate_item_id = items_list[idx]
                            if candidate_item_id in user_ratings['item'].values:
                                continue
                            score_dict[candidate_item_id] = score_dict.get(candidate_item_id, 0.0) + sim * r_val
                            weight_sum[candidate_item_id] = weight_sum.get(candidate_item_id, 0.0) + sim

                    # finalize estimates
                    for cid, ssum in score_dict.items():
                        denom = weight_sum.get(cid, 1e-9)
                        est = ssum / denom
                        users.append(user_id); courses.append(cid); scores.append(float(est))


        elif model_name == models[5]:  # "NMF"
            algo = joblib.load(os.path.join(MODELS_DIR, 'nmf_model.joblib'))
            ratings = load_ratings()
            items_list = ratings['item'].unique()
            unseen = _unseen_items_for_user(user_id, items_list)
            for it in unseen:
                est = algo.predict(uid=user_id, iid=it).est
                users.append(user_id); courses.append(it); scores.append(est)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    res_df = res_df.sort_values('SCORE', ascending=False)
    if "top_courses" in params:
        return res_df.head(params['top_courses'])
    return res_df
