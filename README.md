# FloraVision

FloraVision is a Streamlit flower classification app. It loads a trained CNN model from `cnn_model.keras` and predicts one of five flower classes from an uploaded image.

## Supported Classes

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

## Project Files

- `main.py` - Streamlit app for manual image upload and prediction.
- `cnn_model.keras` - Trained Keras CNN model used by the app.
- `flower_prediction.ipynb` - Notebook used for training and evaluation.
- `flower_images/` - Training image dataset.
- `requirements.txt` - Python dependencies.

## Model File

The app first looks for:

```text
cnn_model.keras
```

Keep this file in the project root if you want the app to work locally or after deployment.

If you add `cnn_model.keras` to `.gitignore`, it will not be uploaded to GitHub. In that case, deployment will only work if you provide a model URL using the `MODEL_URL` environment variable or deployment secret.

Because the current model file is small, the simplest option is to commit `cnn_model.keras` with the project.

## Setup

Create or activate your Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

Use the same Python environment where the dependencies are installed:

```powershell
python -m streamlit run main.py
```

For the local `genai` environment used during testing:

```powershell
$env:PYTHONNOUSERSITE="1"
D:\anaconda3\envs\genai\python.exe -m streamlit run main.py
```

Then open:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit app.
2. Upload a `.jpg`, `.jpeg`, or `.png` flower image.
3. Check the predicted class, confidence score, and probabilities for all classes.

## Deployment Notes

For deployment, use one of these model options:

- Commit `cnn_model.keras` with the repo.
- Upload the model to cloud storage and set `MODEL_URL`.
- Use Git LFS for larger model files.

The app supports direct `.keras` model URLs, zipped TensorFlow SavedModel folders, and Hugging Face model repository URLs.

## Training Notes

The notebook trains a CNN on five flower classes and saves the model as:

```python
model.save("cnn_model.keras")
```

The app expects input images resized to `224x224` RGB and converts model logits to probabilities using softmax.
