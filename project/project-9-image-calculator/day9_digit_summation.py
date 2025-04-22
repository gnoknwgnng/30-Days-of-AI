import streamlit as st
import easyocr
import sympy as sp
import re
from PIL import Image
import numpy as np

# Initialize EasyOCR reader (use CPU if no GPU)
reader = easyocr.Reader(['en'], gpu=False)

# Function to clean and parse extracted text
def parse_expression(text):
    # Join multiple text results and clean
    expression = " ".join(text).strip()
    # Remove non-math characters, keep digits, operators, and decimals
    expression = re.sub(r'[^\d+\-*/.() ]', '', expression)
    try:
        # Evaluate using SymPy for safety and precision
        result = sp.sympify(expression).evalf()
        return expression, float(result)
    except Exception as e:
        return expression, f"Error: Invalid expression ({str(e)})"

# Streamlit interface
st.title("AI Calculator: Image to Calculation")
st.write("Upload an image with a mathematical expression (e.g., '2 + 3 * 4')")

# Image upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array for EasyOCR
    image_np = np.array(image)

    # Extract text using EasyOCR
    with st.spinner("Extracting expression..."):
        results = reader.readtext(image_np, detail=0)  # detail=0 returns only text
        if results:
            expression, result = parse_expression(results)
            st.write(f"Extracted Expression: **{expression}**")
            st.write(f"Result: **{result}**")
        else:
            st.write("No text detected in the image. Try a clearer image.")

# Instructions for users
st.write("Note: Ensure the expression is clear and uses standard math symbols (+, -, *, /, etc.).")