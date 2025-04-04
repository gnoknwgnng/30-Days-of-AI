# Install dependencies first (run in terminal):
# pip install flask nltk

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, request, render_template

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# Initialize Flask app and sentiment analyzer
app = Flask(__name__)
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        if text:
            sentiment, scores = analyze_sentiment(text)
            return render_template('index.html', text=text, sentiment=sentiment, scores=scores)
    return render_template('index.html', text='', sentiment='', scores=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)