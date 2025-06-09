import os
import base64
import pickle
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Path in serverless temp folder
MODEL_PATH = "/tmp/feature_extractor.keras"

# Your public file URL (Google Drive direct download or Hugging Face URL)
FEATURE_EXTRACTOR_URL = "https://drive.google.com/uc?id=1Z73ZnT3v7f3bqzB9jP5W2q5r34IT-VBw"




# Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(FEATURE_EXTRACTOR_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return MODEL_PATH

# Load models
feature_model = load_model(download_model())  # from /tmp
caption_model = load_model("model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34

def extract_features(img):
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_model.predict(x)
    return features

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

def handler(request):
    try:
        body = request.json()
        img_data = base64.b64decode(body['image'])
        img = Image.open(BytesIO(img_data)).convert("RGB")
        features = extract_features(img)
        caption = generate_caption(features)
        return {
            "statusCode": 200,
            "body": {"caption": caption}
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }
