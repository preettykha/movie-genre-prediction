# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
from fuzzywuzzy import process

# Load the dataset
movies = pd.read_csv("C:/Users/Nanthini/Downloads/archive (1)/movies_metadata.csv", low_memory=False)

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

# Create pipelines for vectorizing the titles and training the models
logistic_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

decision_tree_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])

# Define parameter grids for hyper-parameter tuning
logistic_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

decision_tree_param_grid = {
    'classifier__max_depth': [5, 10, 15, 20, 25],
    'classifier__min_samples_split': [2, 5, 10]
}

# Perform grid search for hyper-parameter tuning
logistic_grid_search = GridSearchCV(logistic_pipeline, logistic_param_grid, cv=5, n_jobs=-1)
logistic_grid_search.fit(X_train['title'], y_train)

decision_tree_grid_search = GridSearchCV(decision_tree_pipeline, decision_tree_param_grid, cv=5, n_jobs=-1)
decision_tree_grid_search.fit(X_train['title'], y_train)

# Select the best models
best_logistic_model = logistic_grid_search.best_estimator_
best_decision_tree_model = decision_tree_grid_search.best_estimator_

# Function to find the closest match for a given title in the test set
def find_closest_match(title, choices):
    match = process.extractOne(title, choices)
    return match[0] if match[1] > 75 else None

# Function to predict genres using the best logistic model
def predict_genre_logistic(title):
    closest_title = find_closest_match(title, X_test['title'].tolist())
    if closest_title:
        predicted_genre = best_logistic_model.predict([closest_title])
        return label_encoder.inverse_transform(predicted_genre)[0]
    else:
        return "Movie not found"

# Function to predict genres using the best decision tree model
def predict_genre_decision_tree(title):
    closest_title = find_closest_match(title, X_test['title'].tolist())
    if closest_title:
        predicted_genre = best_decision_tree_model.predict([closest_title])
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

# Genre prediction using logistic regression
st.write("### Predict Movie Genre using Logistic Regression")
movie_title_logistic = st.text_input("Enter the movie title (Logistic Regression):")
if movie_title_logistic:
    genre_logistic = predict_genre_logistic(movie_title_logistic)
    st.write(f"Genres for '{movie_title_logistic}': {genre_logistic}")

# Genre prediction using decision tree
st.write("### Predict Movie Genre using Decision Tree")
movie_title_decision_tree = st.text_input("Enter the movie title (Decision Tree):")
if movie_title_decision_tree:
    genre_decision_tree = predict_genre_decision_tree(movie_title_decision_tree)
    st.write(f"Genres for '{movie_title_decision_tree}': {genre_decision_tree}")

# Display the accuracy of both models
st.write("### Model Accuracy")
y_pred_logistic = best_logistic_model.predict(X_test['title'])
y_pred_decision_tree = best_decision_tree_model.predict(X_test['title'])
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
st.write(f'Accuracy (Logistic Regression): {accuracy_logistic}')
st.write(f'Accuracy (Decision Tree): {accuracy_decision_tree}')

# Display additional evaluation metrics for both models
st.write("### Logistic Regression Evaluation Metrics")
st.write(confusion_matrix(y_test, y_pred_logistic))
st.write(classification_report(y_test, y_pred_logistic, target_names=label_encoder.classes_))

st.write("### Decision Tree Evaluation Metrics")
st.write(confusion_matrix(y_test, y_pred_decision_tree))
st.write(classification_report(y_test, y_pred_decision_tree, target_names=label_encoder.classes_))
