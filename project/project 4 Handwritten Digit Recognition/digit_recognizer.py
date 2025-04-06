import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Load and train the model (or load a pre-trained one)
@st.cache_resource
def load_or_train_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"Model trained! Test accuracy: {test_acc:.4f}")

    return model

# Preprocess uploaded image
def preprocess_image(image):
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Invert colors (MNIST has white digits on black background)
    img_array = 1 - img_array
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Streamlit Interface
def main():
    st.title("Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9) and let the AI predict it!")

    # Load or train the model
    model = load_or_train_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100

        # Show prediction
        st.write(f"**Predicted Digit:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Visualize processed image
        st.write("Processed Image (what the model sees):")
        fig, ax = plt.subplots()
        ax.imshow(processed_image.reshape(28, 28), cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

    # Instructions
    st.write("""
    **Tips:**
    - Draw a digit (0-9) on a white background with a black pen/marker.
    - Ensure the image is clear and centered.
    - Supported formats: PNG, JPG, JPEG.
    """)

if __name__ == "__main__":
    main()