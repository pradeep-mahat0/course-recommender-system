import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()

@st.cache_data
def load_course_genre():
    return backend.load_course_genre()

@st.cache_data
def load_user_profile():
    return backend.load_user_profile()

@st.cache_data
def load_cluster_df():
    return backend.load_cluster_df()

@st.cache_data
def load_cluster_pca():
    return backend.load_cluster_pca()



# Initialize the app by first loading datasets
def init__recommender_app():
    message_placeholder = st.empty()
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # # Select courses
    # st.success('Datasets loaded successfully...')

        # Display the success message in the placeholder
    message_placeholder.success('Datasets loaded successfully...')

    # Pause the script for 7 seconds
    time.sleep(1)

    # Clear the placeholder, which removes the message
    message_placeholder.empty()

    # st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")


    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()


    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        theme="streamlit",   # 👈 Options: "dark", "blue,fresh,light", "material", etc.
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        height=400, 
    )

    

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    

    st.subheader("Your courses: ")
    st.table(results)
    return results

    


def train(model_name, params):

    if model_name == backend.models[0]:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            # backend.train(model_name)
        st.success('Done!')
    # TODO: Add other model training code here
    elif model_name == backend.models[1]:
        with st.spinner('Training...'):
            time.sleep(0.5)
            # backend.train(model_name)
        st.success('Done!')
        
    elif model_name == backend.models[2]:
        
        # Start training KMeans clustering
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name,params)
        st.success('Done!')

    elif model_name == backend.models[3]:
        
        # Start training clustering with PCA
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name,params)
        st.success('Done!')
    

    elif model_name == backend.models[4]:
        
        # knn
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name,params)
        st.success('Done!')

    elif model_name == backend.models[5]:
        with st.spinner('Training NMF...'):
            backend.train(model_name, params)
        st.success('NMF model trained!')

    elif model_name == backend.models[6]: # Neural Network
        # This is the new part for the Neural Network
        with st.spinner('Training Neural Network... This may take a few minutes.'):
            # The backend function will print epoch progress to your console/terminal
            backend.train(model_name, params)
        st.success('Neural Network trained successfully!')

    elif model_name == backend.models[7]: # Regression with Embedding Features
        with st.spinner('Training Regression model...'):
            backend.train(model_name, params)
        st.success('Regression model trained!')

    elif model_name == backend.models[8]: # Classification with Embedding Features
        with st.spinner('Training Classification model...'):
            backend.train(model_name, params)
        st.success('Classification model trained!')
        
    else:
        st.error("Selected model training is not implemented yet.")

    

def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
        
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=10, step=1)
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=50,
                                              value=15, step=5)
    params['top_courses'] = top_courses
    params['profile_sim_threshold']=profile_sim_threshold

# Clustering model
elif model_selection == backend.models[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=30,
                                   value=20, step=1)
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=10, step=1)
    params['cluster_no']=cluster_no
    params['top_courses']=top_courses

# Clustering with PCA 
elif model_selection == backend.models[3]:
    n_components =st.sidebar.slider('Number of Components',
                                    min_value=1,max_value=14,
                                    value=9,step=1)
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=30,
                                   value=20, step=1)
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=20,
                                    value=10, step=1)
    params['cluster_no']=cluster_no
    params['top_courses']=top_courses
    params['n_components']=n_components
    
elif model_selection == backend.models[4]:  # "KNN"
    top_courses = st.sidebar.slider('Top courses', 1, 30, 10, 1)
    k = st.sidebar.slider('k (neighbors)', 5, 100, 40, 5)
    user_based = st.sidebar.radio('Type', ['user-based', 'item-based']) == 'user-based'
    params['top_courses'] = top_courses
    params['k'] = k
    params['user_based'] = user_based

elif model_selection == backend.models[5]:  # "NMF"
    top_courses = st.sidebar.slider('Top courses', 1, 30, 10, 1)
    n_factors = st.sidebar.slider('Latent factors', 10, 200, 50, 10)
    n_epochs = st.sidebar.slider('Epochs', 5, 100, 20, 5)
    reg_pu = st.sidebar.number_input('Reg pu', 0.0, 1.0, 0.06, 0.01)
    reg_qi = st.sidebar.number_input('Reg qi', 0.0, 1.0, 0.06, 0.01)
    params['top_courses'] = top_courses
    params['n_factors'] = n_factors
    params['n_epochs'] = n_epochs
    params['reg_pu'] = reg_pu
    params['reg_qi'] = reg_qi

elif model_selection == backend.models[6]: # "Neural Network"
    st.sidebar.markdown("This model trains embeddings. Prediction uses these embeddings.")
    params['embedding_size'] = st.sidebar.slider('Embedding Size', 8, 32, 16, 4)
    params['epochs'] = st.sidebar.slider('Epochs', 1, 20, 5, 1)
    params['batch_size'] = st.sidebar.slider('Batch Size', 16, 128, 64, 16)
    params['top_courses'] = st.sidebar.slider('Top courses for prediction', 1, 30, 10, 1)

elif model_selection == backend.models[7]: # "Regression with Embedding Features"
    st.sidebar.markdown("Uses embeddings from the NN model to predict ratings.")
    st.sidebar.warning("Note: The 'Neural Network' model must be trained first to generate embeddings.")
    params['top_courses'] = st.sidebar.slider('Top courses', 1, 30, 10, 1)

elif model_selection == backend.models[8]: # "Classification with Embedding Features"
    st.sidebar.markdown("Uses embeddings from the NN model to predict likes/dislikes.")
    st.sidebar.warning("Note: The 'Neural Network' model must be trained first to generate embeddings.")
    params['top_courses'] = st.sidebar.slider('Top courses', 1, 30, 10, 1)
    params['n_estimators'] = st.sidebar.slider('Number of Trees (n_estimators)', 50, 200, 100, 10)
    params['max_depth'] = st.sidebar.slider('Max Depth of Trees', 5, 20, 10, 1)

else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)

    # Check if prediction returned a valid DataFrame
    if not res_df.empty:
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        st.dataframe(res_df)
