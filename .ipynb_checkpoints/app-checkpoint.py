import streamlit as st
import joblib
import numpy as np
import time

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('iris_rf_model.joblib')

model = load_model()

# Image URLs for the 3 species
species_images = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/f/f8/Iris_virginica_2.jpg'
}

st.title("🌸 Iris Flower Classifier (IIT-Level AI)")
st.write("Enter flower measurements to predict its species.")

# Sliders for user input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Placeholder for dynamic image gallery
gallery_placeholder = st.empty()

def show_images(predicted_species=None):
    cols = st.columns(3)
    for idx, (species, img_url) in enumerate(species_images.items()):
        with cols[idx]:
            if predicted_species is None:
                # Default state: Normal images
                st.markdown(f'<div style="text-align:center"><img src="{img_url}" width="100%" style="border-radius:10px;"><br><b>{species.capitalize()}</b></div>', unsafe_allow_html=True)
            elif species == predicted_species:
                # Highlighted state: Correct prediction (Green border + Glow)
                st.markdown(f'<div style="text-align:center"><img src="{img_url}" width="100%" style="border: 4px solid #4CAF50; border-radius:10px; box-shadow: 0px 0px 15px #4CAF50;"><br><b style="color:#4CAF50">{species.capitalize()}</b></div>', unsafe_allow_html=True)
            else:
                # Blurred state: Incorrect predictions
                st.markdown(f'<div style="text-align:center"><img src="{img_url}" width="100%" style="border-radius:10px; filter: blur(4px) grayscale(80%); opacity: 0.4;"><br><b style="color:gray">{species.capitalize()}</b></div>', unsafe_allow_html=True)

# Show normal images initially
with gallery_placeholder.container():
    show_images()

if st.button("Predict Species"):
    # UI Animation effect (Spinner)
    with st.spinner("AI is analyzing the physical characteristics... 🧠"):
        time.sleep(1.5) # 1.5 seconds ka dramatic loading effect

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0].lower()
    confidence = np.max(model.predict_proba(input_data)) * 100

    # Update gallery: blur wrong ones, highlight correct one
    gallery_placeholder.empty() # Clear old images
    with gallery_placeholder.container():
        show_images(prediction)

    st.success(f"Predicted Species: **{prediction.upper()}**")
    st.info(f"AI Confidence Score: **{confidence:.2f}%**")
