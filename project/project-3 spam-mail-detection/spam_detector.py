import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# Load a sample dataset (using 20 newsgroups as a proxy; replace with spam dataset if available)
# Note: For real spam detection, you'd use a dataset like Enron or SMS Spam Collection
categories = ['alt.atheism', 'sci.med']  # Proxy for spam/non-spam
news_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train = news_data.data
y_train = news_data.target  # 0 = alt.atheism (proxy for spam), 1 = sci.med (proxy for not spam)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Custom rule-based features (simple keyword check)
def apply_rules(text):
    spam_keywords = ['free', 'win', 'click', 'money', 'urgent', 'died']  # Add more as needed
    return any(keyword in text.lower() for keyword in spam_keywords)

# Function to classify text
def classify_text(text, threshold=0.5):
    try:
        # Get probability from the model
        prob = model.predict_proba([text])[0]
        score = prob[0]  # Probability of class 0 (spam proxy)
        
        # Combine with rule-based score (simple weighting)
        rule_score = 0.7 if apply_rules(text) else 0.3  # Boost if keywords found
        combined_score = (score + rule_score) / 2  # Average the scores
        
        # Classify based on threshold
        if combined_score >= threshold:
            return "Spam", combined_score
        else:
            return "Not Spam", combined_score
    except Exception as e:
        return "Error", str(e)

# Streamlit interface
def main():
    # Set page title and description
    st.title("Spam Detector")
    st.markdown("""
    Enter an email or text message below to classify it as **Spam** or **Not Spam**.
    Powered by a rule-based system and Naive Bayes classifier.
    """)

    # Text input
    user_input = st.text_area("Input Text", placeholder="Type your message here...", height=150)

    # Threshold slider
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Classify button
    if st.button("Classify"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Classifying..."):
                result, confidence = classify_text(user_input, threshold)
                if result == "Error":
                    st.error(f"An error occurred: {confidence}")
                else:
                    # Display result with color coding
                    if result == "Spam":
                        st.error(f"Result: {result} (Confidence: {confidence:.2f})")
                    else:
                        st.success(f"Result: {result} (Confidence: {confidence:.2f})")

    # Example texts
    st.subheader("Try These Examples")
    examples = [
        "Win a free iPhone now! Click here: http://shady.link",
        "Hey, let’s meet at 3 PM tomorrow.",
        "You have died, claim your prize urgently!"
    ]
    for example in examples:
        if st.button(f"Test: '{example[:30]}...'"):
            result, confidence = classify_text(example, threshold)
            if result == "Error":
                st.error(f"An error occurred: {confidence}")
            else:
                st.write(f"Text: {example}")
                if result == "Spam":
                    st.error(f"Result: {result} (Confidence: {confidence:.2f})")
                else:
                    st.success(f"Result: {result} (Confidence: {confidence:.2f})")

    # Footer
    st.markdown("---")
    st.write("Built with Streamlit and scikit-learn | © 2025")

if __name__ == "__main__":
    main()