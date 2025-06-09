from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import os
import numpy as np
import pickle
import requests

from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = FastAPI()

# === GOOGLE DRIVE FILE IDS (replace with your own) ===
FEATURE_EXTRACTOR_ID = "1Z73ZnT3v7f3bqzB9jP5W2q5r34IT-VBw"
MODEL_ID = "1nUGeJRVN52HjwEUIVrRiloxETRgP_kGi"
TOKENIZER_ID = "1u0lZ-RTN8dt9zf6iQYAHQMn4Ii8RkLx5"

# === DOWNLOAD FUNCTION ===
def download_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url, allow_redirects=True)
    with open(dest_path, 'wb') as f:
        f.write(response.content)

# === DOWNLOAD MODEL FILES AT STARTUP ===
if not os.path.exists("feature_extractor.keras"):
    print("Downloading feature extractor...")
    download_from_drive(FEATURE_EXTRACTOR_ID, "feature_extractor.keras")
if not os.path.exists("model.keras"):
    print("Downloading main model...")
    download_from_drive(MODEL_ID, "model.keras")
if not os.path.exists("tokenizer.pkl"):
    print("Downloading tokenizer...")
    download_from_drive(TOKENIZER_ID, "tokenizer.pkl")

# === LOAD MODELS & TOKENIZER ===
print("Loading models...")
feature_model = load_model("feature_extractor.keras")
caption_model = load_model("model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34

# === FEATURE EXTRACTION ===
def extract_features(img):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_model.predict(x)
    return features

# === CAPTION GENERATION ===
def generate_caption(photo):
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), 'constant')
        sequence = np.expand_dims(sequence, axis=0)
        yhat = caption_model.predict([photo, sequence])
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, "")
        if word == "endseq" or word == "":
            break
        in_text += " " + word
    return in_text.replace("startseq", "").strip()

# === REQUEST BODY FORMAT ===
class ImageData(BaseModel):
    image: str  # base64 string

# === API ENDPOINT ===
@app.post("/caption")
async def caption_endpoint(data: ImageData):
    img_data = base64.b64decode(data.image)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    features = extract_features(img)
    caption = generate_caption(features)
    return {"caption": caption}
