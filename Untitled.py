# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st
from fuzzywuzzy import process

# Load the dataset with a relative path
# Load the dataset
movies = pd.read_csv("movies_metadata.csv", low_memory=False)


# Drop unnecessary columns
columns_to_drop = ['homepage', 'poster_path', 'overview', 'tagline', 'status', 'original_language', 'spoken_languages']
existing_columns_to_drop = [col for col in columns_to_drop if col in movies.columns]
movies = movies.drop(existing_columns_to_drop, axis=1)

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

# Create a pipeline for vectorizing the titles and training the logistic regression model
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train['title'], y_train)

# Function to find the closest match for a given title in the test set
def find_closest_match(title, choices):
    match = process.extractOne(title, choices)
    return match[0] if match[1] > 75 else None

# Function to predict genres
def predict_genre(title):
    closest_title = find_closest_match(title, X_test['title'].tolist())
    if closest_title:
        predicted_genre = pipeline.predict([closest_title])
        return label_encoder.inverse_transform(predicted_genre)[0]
    else:
        return "Movie not found"

# Streamlit part

# Title of the app
st.title('Movie Data Analysis and Genre Prediction')

# Display the dataframe
st.write(movies.head())

# Display the plots
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

# Display the genre distribution
st.write("Genre Distribution")
fig3, ax3 = plt.subplots(figsize=(10, 6))
movies['genre'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_title('Distribution of Genres')
ax3.set_xlabel('Genre')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)

# Display the 100 randomly picked test movies
st.write("### List of Test Movies")
test_movies_df = X_test.head(100)
test_movies_df['genres'] = [movies.loc[movies['title'] == title]['genres'].values[0] for title in test_movies_df['title']]
st.write(test_movies_df)

# Genre prediction
st.write("### Predict Movie Genre")
movie_title = st.text_input("Enter the movie title:")
if movie_title:
    genre = predict_genre(movie_title)
    st.write(f"Genres for '{movie_title}': {genre}")

# Display the accuracy
y_pred = pipeline.predict(X_test['title'])
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy}')





