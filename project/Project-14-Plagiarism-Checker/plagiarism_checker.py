import streamlit as st
from difflib import SequenceMatcher

st.set_page_config(page_title="Plagiarism Checker", layout="centered")
st.title("Plagiarism Checker – Compare Text Similarity")

st.write("""
Paste two pieces of text below to compare their similarity. The similarity score is computed using Python's SequenceMatcher.
""")

text1 = st.text_area("Text 1", height=200)
text2 = st.text_area("Text 2", height=200)

if st.button("Compare"):
    if text1.strip() == "" or text2.strip() == "":
        st.warning("Please enter both texts to compare.")
    else:
        matcher = SequenceMatcher(None, text1, text2)
        similarity = matcher.ratio() * 100
        st.success(f"Similarity Score: {similarity:.2f}%")
        if similarity > 80:
            st.info("High similarity detected! Possible plagiarism.")
        elif similarity > 50:
            st.info("Moderate similarity detected.")
        else:
            st.info("Low similarity detected.")
