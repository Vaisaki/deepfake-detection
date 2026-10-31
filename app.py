
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("Deepfake Detection System")
st.write("Upload an image to check whether it is REAL or FAKE.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_model.keras")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=400)

    if st.button("Detect Deepfake"):
        try:
            model = load_model()

            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array, verbose=0)[0][0]

            if prediction >= 0.5:
                st.success("Prediction: REAL")
                st.metric("Confidence", f"{prediction * 100:.2f}%")
            else:
                st.error("Prediction: FAKE")
                st.metric("Confidence", f"{(1 - prediction) * 100:.2f}%")

            st.warning(
                "This is an experimental demonstration. "
                "Predictions are not reliable proof of authenticity."
            )

        except Exception as e:
            st.error(f"Error: {e}")
