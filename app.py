from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process  # For fuzzy matching

app = Flask(__name__)

# Load dataset
df = pd.read_csv('IMDB-Movie-Data.csv')

@app.route('/')
def index():
    return render_template('index.html', title="Home - Movie Explorer")

@app.route('/visualize/genre')
def visualize_genre():
    genre_avg_rating = df.groupby('Genre')['Rating'].mean()
    
    # Create a figure with more space for labels
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the bar chart and improve the layout
    genre_avg_rating.plot(kind='bar', color='lightblue', ax=ax)
    ax.set_title('Average Rating by Genre', fontsize=16)
    ax.set_xlabel('Genre', fontsize=14)
    ax.set_ylabel('Average Rating', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjust layout to prevent overlap

    # Convert the plot to a base64 string to display in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Use bbox_inches='tight' to avoid cut-off
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    return render_template('visualization.html', title="Genre Visualization", plot_url=plot_url)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name').strip().lower()
        if not movie_name:
            return render_template(
                'recommendation.html',
                title="Movie Recommendation",
                error="Please enter a movie name.",
                recommendations=None
            )
        recommendations, closest_match = content_based_recommendation(movie_name)
        if not recommendations:
            return render_template(
                'recommendation.html',
                title="Movie Recommendation",
                movie_name=movie_name,
                closest_match=closest_match,
                error="Movie not found or no recommendations available.",
                recommendations=None
            )
        return render_template(
            'recommendation.html',
            title="Movie Recommendation",
            movie_name=closest_match,  # Display the matched movie name
            recommendations=recommendations
        )

    return render_template('recommendation.html', title="Movie Recommendation", recommendations=None)

def content_based_recommendation(movie_name):
    """Recommend movies similar to the given movie based on description similarity with fuzzy matching."""
    # Normalize movie titles to lowercase
    df['Title_lower'] = df['Title'].str.lower()

    # Fuzzy matching to find the closest match
    closest_match = process.extractOne(movie_name, df['Title_lower'].values)
    if closest_match is None or closest_match[1] < 70:  # Match confidence threshold
        return [], None

    # Find the index of the closest match
    matched_movie = closest_match[0]
    idx = df[df['Title_lower'] == matched_movie].index[0]

    # Vectorize descriptions and compute cosine similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 excluding itself
    movie_indices = [i[0] for i in sim_scores]

    return df.iloc[movie_indices][['Title', 'Genre', 'Rating']].to_dict('records'), matched_movie

if __name__ == '__main__':
    app.run(debug=True)
