import os
import site
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

for user_site_path in {site.getusersitepackages()}:
    sys.path = [
        path for path in sys.path
        if os.path.normcase(path) != os.path.normcase(user_site_path)
    ]

import streamlit as st
import tensorflow as tf
import numpy as np
import zipfile
from PIL import Image
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
MODEL_PATH = Path("models") / "flower_classifier"
LOCAL_KERAS_MODEL_PATH = Path("cnn_model.keras")
DOWNLOADED_MODEL_PATH = Path("models") / "downloaded_model.keras"
DOWNLOADED_ZIP_PATH = Path("models") / "downloaded_model.zip"
HF_MODEL_PATH = Path("models") / "hf_model"


def load_dotenv_file():
    """Load simple KEY=VALUE entries from .env without requiring python-dotenv."""
    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file()

# Configure Streamlit page
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)
UNKNOWN_THRESHOLD = 0.45

def get_model_url():
    """Read model URL from Streamlit secrets or environment variables."""
    try:
        model_url = st.secrets.get("MODEL_URL") or os.getenv("MODEL_URL")
    except Exception:
        model_url = os.getenv("MODEL_URL")

    if not model_url:
        return None

    # Hugging Face direct downloads use /resolve/, while copied browser URLs often use /blob/.
    return model_url.replace("/blob/", "/resolve/")


def parse_huggingface_url(model_url):
    """Return repo information for Hugging Face model URLs."""
    parsed = urlparse(model_url)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        return None

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None

    repo_id = "/".join(parts[:2])
    folder = None
    revision = "main"

    if len(parts) >= 4 and parts[2] in {"tree", "resolve", "blob"}:
        revision = parts[3]
        if parts[2] == "tree" and len(parts) > 4:
            folder = "/".join(parts[4:])

    return repo_id, revision, folder


def find_saved_model_dir(base_path):
    """Find a TensorFlow SavedModel folder under base_path."""
    base_path = Path(base_path)
    if (base_path / "saved_model.pb").exists():
        return base_path

    saved_model_files = list(base_path.rglob("saved_model.pb"))
    if saved_model_files:
        return saved_model_files[0].parent

    return None


def download_huggingface_model(model_url):
    """Download a Hugging Face model repo or folder and return the local model path."""
    parsed = parse_huggingface_url(model_url)
    if parsed is None:
        return None

    repo_id, revision, folder = parsed

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        st.error("huggingface_hub is required for Hugging Face repo URLs.")
        st.info("Install dependencies again after updating requirements.txt.")
        return None

    allow_patterns = [f"{folder}/**"] if folder else None
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(HF_MODEL_PATH),
        allow_patterns=allow_patterns,
    )

    saved_model_dir = find_saved_model_dir(snapshot_path)
    if saved_model_dir:
        return saved_model_dir

    keras_files = list(Path(snapshot_path).rglob("*.keras"))
    if keras_files:
        return keras_files[0]

    return Path(snapshot_path)

def ensure_model_file():
    """Use local model if available, otherwise download it for deployment."""
    if LOCAL_KERAS_MODEL_PATH.exists():
        return LOCAL_KERAS_MODEL_PATH

    if MODEL_PATH.exists():
        return MODEL_PATH

    for legacy_model_path in (
        DOWNLOADED_MODEL_PATH,
        Path("shallow_model.keras"),
        Path("models") / "shallow_model.keras",
    ):
        if legacy_model_path.exists():
            return legacy_model_path

    model_url = get_model_url()
    if not model_url:
        st.error("Model file was not found.")
        st.info("Train the notebook to create cnn_model.keras, or set MODEL_URL in your deployment secrets.")
        return None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model..."):
        hf_model_path = download_huggingface_model(model_url)
        if hf_model_path is not None:
            return hf_model_path

        if ".zip" in model_url.lower():
            urlretrieve(model_url, DOWNLOADED_ZIP_PATH)
            with zipfile.ZipFile(DOWNLOADED_ZIP_PATH) as archive:
                archive.extractall(MODEL_PATH)

            saved_model_dir = find_saved_model_dir(MODEL_PATH)
            if saved_model_dir:
                return saved_model_dir
            return MODEL_PATH

        urlretrieve(model_url, DOWNLOADED_MODEL_PATH)

    return DOWNLOADED_MODEL_PATH


class SavedModelPredictor:
    """Keras 3 wrapper for TensorFlow SavedModel inference exports."""

    def __init__(self, model_path):
        self.layer = tf.keras.layers.TFSMLayer(
            str(model_path),
            call_endpoint="serving_default",
        )

    def predict(self, image, verbose=0):
        outputs = self.layer(tf.convert_to_tensor(image, dtype=tf.float32), training=False)
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        return outputs.numpy()

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the trained flower classification model"""
    model_path = ensure_model_file()
    if model_path is None:
        return None

    try:
        if model_path.is_dir() and (model_path / "saved_model.pb").exists():
            if hasattr(tf.keras.layers, "TFSMLayer"):
                return SavedModelPredictor(model_path)
            return tf.keras.models.load_model(model_path)

        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Make sure h5py is installed and MODEL_URL points to a Hugging Face model repo, a direct .keras file, or a zipped TensorFlow SavedModel.")
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
            st.image(image, caption="Your flower image", width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            processed_image = preprocess_image(image)
            
            with st.spinner("Analyzing the flower..."):
                predicted_class, confidence, probabilities = predict_flower(model, processed_image)
            
            if predicted_class is not None:
                confidence_label = (
                    "High" if confidence >= 0.70
                    else "Moderate" if confidence >= UNKNOWN_THRESHOLD
                    else "Low"
                )

                if confidence < UNKNOWN_THRESHOLD:
                    st.warning(f"**Top prediction: {predicted_class.title()}**")
                    st.write(f"**Confidence: {confidence:.1%}** ({confidence_label}; threshold {UNKNOWN_THRESHOLD:.1%})")
                    st.write("The model is not confident enough to make a reliable prediction.")
                else:
                    st.success(f"**Prediction: {predicted_class.title()}**")
                    st.write(f"**Confidence: {confidence:.1%}** ({confidence_label})")

                    if confidence > 0.7:
                        st.success("High confidence prediction")
                    elif confidence > UNKNOWN_THRESHOLD:
                        st.warning("Moderate confidence prediction")
                    else:
                        st.error("Low confidence prediction")

                st.subheader("All Class Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = float(probabilities[i])
                    st.write(f"**{class_name.title()}:** {prob:.1%}")
                    st.progress(prob)
            
        st.subheader("📋 Model Information")
        st.info("""
        This model is a convolutional neural network with:
        - **Architecture:** Data augmentation, normalization, convolution, pooling, dropout, and dense classification layers
        - **Input:** 224x224x3 RGB images
        - **Classes:** Daisy, Dandelion, Rose, Sunflower, Tulip
        - **Output:** Class logits converted to probabilities in the app
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
