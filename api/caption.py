from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = FastAPI()

# Load models and tokenizer once
feature_model = load_model("feature_extractor.keras")
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

class ImageData(BaseModel):
    image: str  # base64 string

@app.post("/caption")
async def caption_endpoint(data: ImageData):
    img_data = base64.b64decode(data.image)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    features = extract_features(img)
    caption = generate_caption(features)
    return {"caption": caption}
