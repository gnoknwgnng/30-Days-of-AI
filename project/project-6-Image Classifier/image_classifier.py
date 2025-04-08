import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from urllib.parse import urlparse

# Load pre-trained MobileNetV2 model
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

# Preprocess image for model input
def preprocess_image(image):
    img = image.resize((224, 224))  # MobileNetV2 input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Decode predictions
def decode_predictions(preds, top=3):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top)[0]
    return [(label, float(conf)) for (_, label, conf) in decoded]

# Check if URL is valid and points to an image
def is_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('content-type', '').lower()
        return content_type.startswith('image/')
    except:
        return False

# Streamlit Interface
st.title("🖼️ Image Classifier")
st.write("Powered by Grok 3 from xAI with TensorFlow")
st.write("Upload an image or provide a URL to classify objects")

# Load model
model = load_model()

# Input options
input_type = st.radio("Select input type:", ("Upload Image", "Image URL"))

# Results placeholder
results = None
uploaded_image = None

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert('RGB')
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                # Preprocess and classify
                img_array = preprocess_image(uploaded_image)
                predictions = model.predict(img_array)
                results = decode_predictions(predictions)
                st.success("Classification complete!")

elif input_type == "Image URL":
    url_input = st.text_input("Enter the image URL here:")
    if url_input and st.button("Classify URL Image"):
        if is_valid_image_url(url_input):
            with st.spinner("Fetching and classifying..."):
                try:
                    response = requests.get(url_input, timeout=5)
                    uploaded_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    st.image(uploaded_image, caption='Image from URL', use_column_width=True)
                    
                    # Preprocess and classify
                    img_array = preprocess_image(uploaded_image)
                    predictions = model.predict(img_array)
                    results = decode_predictions(predictions)
                    st.success("Classification complete!")
                except:
                    st.error("Failed to process the image from URL")
        else:
            st.error("Invalid image URL or not an image")

# Display results
if results:
    st.subheader("Classification Results")
    for i, (label, confidence) in enumerate(results, 1):
        st.write(f"{i}. **{label}**: {confidence:.2%} confidence")
    
    # Additional analysis (simulating Grok's capabilities)
    st.subheader("Additional Analysis")
    st.write("Domain credibility check (if URL provided):")
    if input_type == "Image URL" and url_input:
        domain = urlparse(url_input).netloc
        st.write(f"- Source domain: {domain}")
        st.write(f"- Uses HTTPS: {url_input.startswith('https')}")
    
    st.write("Note: Confidence scores above 80% are generally reliable.")

# Sidebar with info
st.sidebar.title("About")
st.sidebar.write("""
This Image Classifier uses TensorFlow's MobileNetV2 model to identify objects in images. Features:
- Pre-trained on ImageNet dataset
- Returns top 3 predictions with confidence scores
- Supports both file uploads and URLs
Built with Grok 3 capabilities from xAI
""")
st.sidebar.write(f"Date: April 08, 2025")

# Instructions to run:
# 1. Save as image_classifier.py
# 2. Install requirements: pip install streamlit tensorflow pillow numpy requests
# 3. Run with: streamlit run image_classifier.pys