# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from fuzzywuzzy import process

# Set Streamlit page configuration
st.set_page_config(page_title="Movie Genre Prediction", page_icon="🎬")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    columns_to_drop = ['homepage', 'poster_path', 'overview', 'tagline', 'status', 'original_language', 'spoken_languages']
    movies = movies.drop(columns=columns_to_drop, axis=1)
    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
    movies['budget'] = pd.to_numeric(movies['budget'].str.replace(',', ''), errors='coerce')
    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')
    movies['runtime'] = pd.to_numeric(movies['runtime'], errors='coerce')
    movies = movies.dropna(subset=['popularity', 'budget', 'revenue', 'runtime', 'vote_average'])
    movies['genres'] = movies['genres'].apply(lambda x: [genre['name'] for genre in eval(x)] if pd.notna(x) else [])
    movies = movies[['title', 'genres', 'popularity', 'budget', 'revenue', 'runtime', 'vote_average']]
    movies['genre'] = movies['genres'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    return movies

movies = load_data()

# Encode the genre labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(movies['genre'])
X_train, X_test, y_train, y_test = train_test_split(movies[['title']], y, test_size=0.2, random_state=42)

# Create pipelines
pipeline_logistic = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])
pipeline_tree = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

@st.cache_resource
def train_models():
    pipeline_logistic.fit(X_train['title'], y_train)
    pipeline_tree.fit(X_train['title'], y_train)
    return pipeline_logistic, pipeline_tree

pipeline_logistic, pipeline_tree = train_models()

def find_closest_match(title, choices):
    match = process.extractOne(title, choices)
    return match[0] if match[1] > 75 else None

def predict_genre(title, model):
    closest_title = find_closest_match(title, X_test['title'].tolist())
    if closest_title:
        predicted_genre = model.predict([closest_title])
        return label_encoder.inverse_transform(predicted_genre)[0]
    else:
        return "Movie not found"

# Streamlit part
st.title('Movie Data Analysis and Genre Prediction')
st.write("### Sample of Movies Data")
st.write(movies.head())

st.write("### List of Test Movies")
test_movies_df = X_test.head(100).copy()
test_movies_df['genres'] = [movies.loc[movies['title'] == title]['genres'].values[0] for title in test_movies_df['title']]
st.write(test_movies_df)

st.write("### Predict Movie Genre")
movie_title = st.text_input("Enter the movie title:")
if movie_title:
    genre_logistic = predict_genre(movie_title, pipeline_logistic)
    genre_tree = predict_genre(movie_title, pipeline_tree)
    st.write(f"Genres for '{movie_title}' using Logistic Regression: {genre_logistic}")
    st.write(f"Genres for '{movie_title}' using Decision Tree: {genre_tree}")

y_pred_logistic = pipeline_logistic.predict(X_test['title'])
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
st.write(f'Logistic Regression Accuracy: {accuracy_logistic}')

y_pred_tree = pipeline_tree.predict(X_test['title'])
accuracy_tree = accuracy_score(y_test, y_pred_tree)
st.write(f'Decision Tree Accuracy: {accuracy_tree}')

st.write("### Visualizations")
st.write("Histogram of Popularity")
fig1, ax1 = plt.subplots(figsize=(10, 6))
movies['popularity'].hist(bins=50, ax=ax1)
ax1.set_title('Histogram of Popularity')
ax1.set_xlabel('Popularity')
ax1.set_ylabel('Frequency')
st.pyplot(fig1)

st.write("Scatter plot of Popularity vs. Vote Average")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(movies['popularity'], movies['vote_average'])
ax2.set_title('Scatter plot of Popularity vs. Vote Average')
ax2.set_xlabel('Popularity')
ax2.set_ylabel('Vote Average')
st.pyplot(fig2)

st.write("Genre Distribution")
fig3, ax3 = plt.subplots(figsize=(10, 6))
movies['genre'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_title('Distribution of Genres')
ax3.set_xlabel('Genre')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

st.write("Confusion Matrix - Logistic Regression")
fig4, ax4 = plt.subplots(figsize=(10, 6))
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
cax4 = ax4.matshow(conf_matrix_logistic, cmap='coolwarm')
fig4.colorbar(cax4)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig4)

st.write("Confusion Matrix - Decision Tree")
fig5, ax5 = plt.subplots(figsize=(10, 6))
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
cax5 = ax5.matshow(conf_matrix_tree, cmap='coolwarm')
fig5.colorbar(cax5)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig5)

# Streamlit custom styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>input {
        background-color: #FFFFFF;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True
)

