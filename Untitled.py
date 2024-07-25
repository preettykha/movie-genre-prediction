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
from fuzzywuzzy import fuzz, process

# Set Streamlit page configuration
st.set_page_config(page_title="Movie Genre Prediction", page_icon="🎬")

# Load the dataset
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    return movies

movies = load_data()

# Drop unnecessary columns
columns_to_drop = ['homepage', 'poster_path', 'overview', 'tagline', 'status', 'original_language', 'spoken_languages']
movies = movies.drop(columns=columns_to_drop, axis=1)

# Ensure all numeric columns are correctly typed
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
movies['budget'] = pd.to_numeric(movies['budget'].str.replace(',', ''), errors='coerce')
movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')
movies['runtime'] = pd.to_numeric(movies['runtime'], errors='coerce')

# Drop rows with NaN values
movies = movies.dropna(subset=['popularity', 'budget', 'revenue', 'runtime', 'vote_average'])

# Extract genres and titles
movies['genres'] = movies['genres'].apply(lambda x: [genre['name'] for genre in eval(x)] if pd.notna(x) else [])
movies = movies[['title', 'genres', 'popularity', 'budget', 'revenue', 'runtime', 'vote_average']]

# Use the first genre as the target variable
movies['genre'] = movies['genres'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')

# Encode the genre labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(movies['genre'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(movies[['title']], y, test_size=0.2, random_state=42)

# Create pipelines for both models
pipeline_logistic = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline_tree = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the models
pipeline_logistic.fit(X_train['title'], y_train)
pipeline_tree.fit(X_train['title'], y_train)

# Function to find the closest match for a given title in the test set
def find_closest_match(title, choices):
    match = process.extractOne(title, choices, scorer=fuzz.ratio)
    return match[0] if match[1] > 75 else None

# Function to predict genres
def predict_genre(title, model):
    closest_title = find_closest_match(title, X_test['title'].tolist())
    if closest_title:
        predicted_genre = model.predict([closest_title])
        return label_encoder.inverse_transform(predicted_genre)[0]
    else:
        return "Movie not found"

# Streamlit part

# Title of the app
st.title('Movie Data Analysis and Genre Prediction')

# Display the dataframe
st.write("### Sample of Movies Data")
st.write(movies.head())

# Display the 100 randomly picked test movies
st.write("### List of Test Movies")
test_movies_df = X_test.head(100).copy()  # Make a copy to avoid the warning
test_movies_df['genres'] = [movies.loc[movies['title'] == title, 'genres'].values[0] for title in test_movies_df['title']]
st.write(test_movies_df)

# Genre prediction
st.write("### Predict Movie Genre")
movie_title = st.text_input("Enter the movie title:", key="movie_title", autocomplete="on")
if movie_title:
    genre_logistic = predict_genre(movie_title, pipeline_logistic)
    genre_tree = predict_genre(movie_title, pipeline_tree)
    st.write(f"Genres for '{movie_title}' using Logistic Regression: {genre_logistic}")
    st.write(f"Genres for '{movie_title}' using Decision Tree: {genre_tree}")

# Display the accuracy for both models
y_pred_logistic = pipeline_logistic.predict(X_test['title'])
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
st.write(f'Logistic Regression Accuracy: {accuracy_logistic}')

y_pred_tree = pipeline_tree.predict(X_test['title'])
accuracy_tree = accuracy_score(y_test, y_pred_tree)
st.write(f'Decision Tree Accuracy: {accuracy_tree}')

# Confusion matrices for both models
st.write("### Confusion Matrices")
fig4, ax4 = plt.subplots(figsize=(10, 6))
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
ax4.matshow(cm_logistic, cmap='coolwarm')
ax4.set_title('Confusion Matrix - Logistic Regression')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(10, 6))
cm_tree = confusion_matrix(y_test, y_pred_tree)
ax5.matshow(cm_tree, cmap='coolwarm')
ax5.set_title('Confusion Matrix - Decision Tree')
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')
st.pyplot(fig5)

# Visualize the results
st.write("### Visualizations")

# Histogram of 'popularity'
st.write("Histogram of Popularity")
fig1, ax1 = plt.subplots(figsize=(10, 6))
movies['popularity'].hist(bins=50, ax=ax1)
ax1.set_title('Histogram of Popularity')
ax1.set_xlabel('Popularity')
ax1.set_ylabel('Frequency')
st.pyplot(fig1)

# Scatter plot of 'popularity' vs 'vote_average'
st.write("Scatter plot of Popularity vs. Vote Average")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(movies['popularity'], movies['vote_average'])
ax2.set_title('Scatter plot of Popularity vs. Vote Average')
ax2.set_xlabel('Popularity')
ax2.set_ylabel('Vote Average')
st.pyplot(fig2)

# Display the genre distribution
st.write("Genre Distribution")
fig3, ax3 = plt.subplots(figsize=(10, 6))
movies['genre'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_title('Distribution of Genres')
ax3.set_xlabel('Genre')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# Streamlit custom styling
st.markdown("""
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
""", unsafe_allow_html=True)
