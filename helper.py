import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  
import joblib
from sklearn.decomposition import PCA



def create_user_profile(user_id, ratings_df, course_genres_df):
    """
    Creates a user profile vector by aggregating a user's ratings
    and mapping them to course genres.
    
    Args:
        user_id (int): The ID of the user to process.
        ratings_df (pd.DataFrame): DataFrame containing user ratings.
        course_genres_df (pd.DataFrame): DataFrame with course genre information.
        
    Returns:
        np.ndarray: A 1D NumPy array representing the user's genre profile.
    """
    # 1. Filter ratings for the specific user
    user_ratings = ratings_df[ratings_df['user'] == user_id]

    # 2. Merge user ratings with course genres on the item ID
    # Use 'left_on' and 'right_on' for clarity
    merged_df = pd.merge(
        user_ratings,
        course_genres_df,
        left_on='item',
        right_on='COURSE_ID',
        how='left'
    )
    
    # 3. Drop unnecessary columns in a single, clean step
    merged_df.drop(
        columns=['COURSE_ID', 'TITLE'],
        inplace=True
    )
    
    # 4. Identify the genre columns
    # This is more robust than hardcoding column names
    excluded_cols = ['user', 'item', 'rating']
    genre_cols = merged_df.columns[~merged_df.columns.isin(excluded_cols)]
    
    # 5. Multiply ratings with genre columns to create weighted ratings
    weighted_ratings_df = merged_df[genre_cols].multiply(merged_df['rating'], axis=0)

    # 6. Aggregate the weighted ratings by user to create a single profile vector
    # The 'reset_index' isn't needed here since we want the values, not the DataFrame
    user_profile_vector = weighted_ratings_df.sum().values
    
    return user_profile_vector

def combine_cluster_labels(user_ids, labels):
    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(labels)    
    # Merge user_ids DataFrame with labels DataFrame based on index
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    # Rename columns to 'user' and 'cluster'
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

def preprocess_and_train_kMeans(user_profile_df,n_clusters):
    #preprocess
    feature_name=list(user_profile_df.columns[1:])
    scaler=StandardScaler()

    user_profile_df[feature_name]=scaler.fit_transform(user_profile_df[feature_name])
    features=user_profile_df.loc[:,user_profile_df.columns !='user']
    user_ids=user_profile_df.loc[:,user_profile_df.columns =='user']

    #model train
    cluster_labels=[None] *len(user_ids)
    model=KMeans(n_clusters,random_state=5)
    model.fit(features)
    cluster_labels=model.labels_
    cluster_df=combine_cluster_labels(user_ids,cluster_labels)
    
    joblib.dump(scaler,'scaler.joblib')
    joblib.dump(model,'KMeans_model.joblib')

    return cluster_df

def predict_cluster(scaler,model,user_profile_df):
    feature_name=list(user_profile_df.columns[1:])
    user_profile_df[feature_name]=scaler.transform(user_profile_df[feature_name])
    features=user_profile_df.loc[:,user_profile_df.columns !='user']
    cluster=model.predict(features)
    return cluster


def pca_and_train(user_profile_df,n_components,n_clusters):

    #preprocess
    feature_name=list(user_profile_df.columns[1:])
    scaler=StandardScaler()

    user_profile_df[feature_name]=scaler.fit_transform(user_profile_df[feature_name])
    features=user_profile_df.loc[:,user_profile_df.columns !='user']
    #pca model
    pca=PCA(n_components=n_components,random_state=5)
    components=pca.fit_transform(features)
    pca_df=pd.DataFrame(data=components,
                        columns=[f"PC{i+1}" for i in range(n_components)])
    
    #clustering with kmeans
    pca_features=pca_df.values
    model=KMeans(n_clusters=n_clusters,random_state=5)
    model.fit(pca_features)

    cluster_labels=model.labels_

    clustered_users = pd.DataFrame({
        "user": user_profile_df["user"],
        "cluster": cluster_labels
    })
    joblib.dump(scaler,'scaler.joblib')
    joblib.dump(pca,'pca_model.joblib')
    joblib.dump(model,'KMeans_with_pca.joblib')

    return clustered_users

def predict_pca_cluster(scaler,pca_model,model,user_profile_df):
    feature_name=list(user_profile_df.columns[1:])
    user_profile_df[feature_name]=scaler.transform(user_profile_df[feature_name])
    features=user_profile_df.loc[:,user_profile_df.columns !='user']

    #pca model
    components=pca_model.transform(features)
    cluster=model.predict(components)
    return cluster
    



    