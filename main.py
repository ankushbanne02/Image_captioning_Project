import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import os

# Function to generate and return caption
def generate_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    try:
        caption_model = load_model(model_path)
        feature_extractor = load_model(feature_extractor_path)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        img = load_img(image_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        image_features = feature_extractor.predict(img_array, verbose=0)

        in_text = "startseq"
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            yhat_index = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat_index)
            if word is None or word == "endseq":
                break
            in_text += " " + word

        caption = in_text.replace("startseq", "").replace("endseq", "").strip()
        return caption, img
    except Exception as e:
        return f"Error: {e}", None


# Streamlit app interface
def main():
    st.set_page_config(page_title="🖼️ Image Caption Generator", layout="centered")

    # Header with styling
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #3366cc;">📸 Image Caption Generator</h1>
            <p style="font-size: 18px;">Upload an image and generate a human-like caption using a deep learning model.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with st.spinner("Generating caption... Please wait."):
            temp_img_path = "uploaded_image.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Paths to the trained models and tokenizer
            model_path = "models/model.keras"
            tokenizer_path = "models/tokenizer.pkl"
            feature_extractor_path = "models/feature_extractor.keras"

            caption, img = generate_caption(temp_img_path, model_path, tokenizer_path, feature_extractor_path)

        if img is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.markdown("### 📝 Generated Caption")
                st.success(caption)
        else:
            st.error(caption)

    st.markdown("---")
    st.markdown(
        "<small style='text-align: center;'>Built with ❤️ using TensorFlow + Streamlit</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
