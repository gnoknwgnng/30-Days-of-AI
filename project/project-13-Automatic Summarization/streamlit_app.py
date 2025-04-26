import streamlit as st
from transformers import pipeline

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline('summarization', model='facebook/bart-large-cnn')

summarizer = load_summarizer()

st.title('Automatic Article Summarizer 📑')
st.write('Paste your article below and get a concise summary using state-of-the-art NLP!')

article_text = st.text_area('Paste Article Text', height=300)

if st.button('Summarize'):
    if not article_text.strip():
        st.warning('Please paste some article text to summarize.')
    else:
        with st.spinner('Summarizing...'):
            # Hugging Face models have a max token limit; chunk if needed
            try:
                summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                st.subheader('Summary')
                st.success(summary)
            except Exception as e:
                st.error(f'Error during summarization: {e}')
