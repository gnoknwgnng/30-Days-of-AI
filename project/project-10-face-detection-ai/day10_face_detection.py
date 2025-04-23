import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="AI Face Detection", page_icon="📸", layout="centered")

# Load Haar Cascade for face detection
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

# Function to detect faces
def detect_faces(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image, len(faces)

# Streamlit app
st.title("📸 AI Face Detection")
st.write("Upload an image to detect faces using AI. Built for #30DaysOfAI!")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and convert image
        image = np.array(Image.open(uploaded_file))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        result, num_faces = detect_faces(image.copy())
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.open(uploaded_file), caption="Original Image", use_column_width=True)
        with col2:
            st.image(result, caption="Faces Detected", use_column_width=True)
        
        # Show number of faces detected
        st.write(f"**Number of faces detected**: {num_faces}")
        
        # Add a fun touch
        if num_faces == 0:
            st.warning("No faces found! Try another image.")
        else:
            st.success(f"Found {num_faces} face{'s' if num_faces > 1 else ''}!")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to get started.")

# Footer
st.markdown("---")
st.markdown("Built by Akash & Team for #30DaysOfAI | Powered by OpenCV & Streamlit")