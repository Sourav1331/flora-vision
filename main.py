import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from pathlib import Path
from urllib.request import urlretrieve

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["daisy", "dandelions", "roses", "sunflowers", "tulip"]
MODEL_PATH = Path("models") / "shallow_model.keras"

# Configure Streamlit page
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)
UNKNOWN_THRESHOLD = 0.70

def get_model_url():
    """Read model URL from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get("MODEL_URL") or os.getenv("MODEL_URL")
    except Exception:
        return os.getenv("MODEL_URL")

def ensure_model_file():
    """Use local model if available, otherwise download it for deployment."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    legacy_model_path = Path("shallow_model.keras")
    if legacy_model_path.exists():
        return legacy_model_path

    model_url = get_model_url()
    if not model_url:
        st.error("Model file was not found.")
        st.info("Add shallow_model.keras locally, or set MODEL_URL in your deployment secrets.")
        return None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model..."):
        urlretrieve(model_url, MODEL_PATH)

    return MODEL_PATH

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the trained flower classification model"""
    model_path = ensure_model_file()
    if model_path is None:
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Make sure the model file exists locally or MODEL_URL points to a valid .keras file.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to model input size
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_flower(model, image):
    """Make prediction on the preprocessed image"""
    try:
        predictions = model.predict(image, verbose=0)
        
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        return predicted_class, confidence, probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    st.title("🌸 Flower Classification App")
    st.write("Upload an image of a flower and I'll predict what type it is!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a flower image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a flower for best results"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Your flower image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            processed_image = preprocess_image(image)
            
            with st.spinner("Analyzing the flower..."):
                predicted_class, confidence, probabilities = predict_flower(model, processed_image)
            
            if predicted_class is not None:

                if confidence < UNKNOWN_THRESHOLD:
                    st.error("Does not seem to be a given flower(out of the 5 classes)")
                    st.write(f"Top model confidence: **{confidence:.1%}** (threshold {UNKNOWN_THRESHOLD:.1%})")
                else:
                    st.success(f"**Prediction: {predicted_class.title()}**")
                    st.write(f"**Confidence: {confidence:.1%}**")
                    
                    if confidence > 0.7:
                        st.success("🎯 High confidence prediction!")
                    elif confidence > 0.4:
                        st.warning("⚠️ Moderate confidence prediction")
                    else:
                        st.error("❌ Low confidence prediction")
                    
                    st.subheader("All Class Probabilities")
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = float(probabilities[i])  # Convert to Python float
                        st.write(f"**{class_name.title()}:** {prob:.1%}")
                        st.progress(prob)
            
        st.subheader("📋 Model Information")
        st.info("""
        This model is a simple neural network with:
        - **Architecture:** Single dense layer (no hidden layers)
        - **Input:** 224x224x3 RGB images
        - **Classes:** Daisy, Dandelions, Roses, Sunflowers, Tulip
        - **Note:** This is a shallow network without activation functions for educational purposes
        """)
        
        st.subheader("💡 Tips for Better Results")
        st.write("""
        - Use clear, well-lit images
        - Make sure the flower is the main subject
        - Try different angles if the prediction confidence is low
        - The model works best with the 5 flower types it was trained on
        """)
    
    else:
        # Show example images and instructions
        st.subheader("📸 How to Use")
        st.write("""
        1. Click 'Browse files' above to upload a flower image
        2. The model will analyze your image and predict the flower type
        3. You'll see the prediction confidence and probabilities for all classes
        """)
        
        # Display supported flower types
        st.subheader("🌼 Supported Flower Types")
        cols = st.columns(5)
        for i, flower in enumerate(CLASS_NAMES):
            with cols[i]:
                st.write(f"**{flower.title()}**")

if __name__ == "__main__":
    main()
