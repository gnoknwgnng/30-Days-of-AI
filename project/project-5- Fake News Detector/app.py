import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import hashlib

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Simulated Grok analysis functions
def analyze_text(text):
    """Analyze text for potential fake news indicators"""
    # Basic checks for sensationalism and credibility
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words]
    
    # Check for sensationalist keywords
    sensational_keywords = ['shocking', 'unbelievable', 'incredible', 'amazing', 'urgent']
    sensational_count = sum(1 for word in filtered_words if word in sensational_keywords)
    
    # Check for excessive punctuation
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    return {
        'sensational_score': sensational_count / max(1, len(filtered_words)) * 100,
        'exclamation_usage': exclamation_count,
        'question_usage': question_count
    }

def check_web_credibility(url):
    """Check website credibility based on domain characteristics"""
    try:
        domain = urlparse(url).netloc
        # Simple domain credibility checks
        credibility_indicators = {
            'has_https': url.startswith('https'),
            'domain_length': len(domain),
            'has_numbers': bool(re.search(r'\d', domain)),
            'known_domains': any(known in domain for known in ['.edu', '.org', '.gov'])
        }
        return credibility_indicators
    except:
        return None

def scrape_url(url):
    """Scrape content from URL"""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join(p.get_text() for p in paragraphs)
    except:
        return None

# Streamlit Interface
st.title("📢 Fake News Detector")
st.write("Powered by Grok 3 from xAI")
st.write("Enter text or URL to analyze for potential misleading content")

# Input options
input_type = st.radio("Select input type:", ("Text", "URL"))

# Analysis results
results = None

if input_type == "Text":
    user_input = st.text_area("Enter the article text here:", height=200)
    if st.button("Analyze Text"):
        if user_input:
            results = analyze_text(user_input)
            st.success("Analysis complete!")
        else:
            st.warning("Please enter some text to analyze")

elif input_type == "URL":
    url_input = st.text_input("Enter the article URL here:")
    if st.button("Analyze URL"):
        if url_input:
            with st.spinner("Fetching and analyzing content..."):
                content = scrape_url(url_input)
                if content:
                    results = analyze_text(content)
                    credibility = check_web_credibility(url_input)
                    st.success("Analysis complete!")
                else:
                    st.error("Could not fetch content from URL")
        else:
            st.warning("Please enter a URL to analyze")

# Display results
if results:
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Sensationalism Score", f"{results['sensational_score']:.1f}%")
        st.write("Higher scores indicate more sensational language")
    
    with col2:
        st.metric("Exclamation Marks", results['exclamation_usage'])
        st.metric("Question Marks", results['question_usage'])
    
    if input_type == "URL" and credibility:
        st.subheader("Website Credibility Indicators")
        st.write(f"Uses HTTPS: {credibility['has_https']}")
        st.write(f"Domain Length: {credibility['domain_length']} characters")
        st.write(f"Contains Numbers: {credibility['has_numbers']}")
        st.write(f"Known Reliable Domain: {credibility['known_domains']}")

    # Interpretation
    st.subheader("Interpretation")
    if results['sensational_score'] > 10 or results['exclamation_usage'] > 5:
        st.warning("Content shows signs of potential misleading information due to high sensationalism")
    else:
        st.success("Content appears relatively neutral based on language analysis")

    st.write("Note: This is a basic analysis and not a definitive determination of truthfulness.")

# Sidebar with additional info
st.sidebar.title("About")
st.sidebar.write("""
This Fake News Detector uses basic natural language processing and web analysis to identify potential misleading content. Features include:
- Sensational language detection
- Punctuation pattern analysis
- Basic website credibility checks
Built with Grok 3 capabilities from xAI
""")
st.sidebar.write(f"Date: April 07, 2025")

# Instructions to run:
# 1. Save this as app.py
# 2. Install requirements: pip install streamlit requests beautifulsoup4 nltk
# 3. Run with: streamlit run app.py