import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_utils import MODEL_PATH, CLASS_NAMES, load_fashion_mnist

st.set_page_config(page_title="Fashion-MNIST CNN Classifier", page_icon="🧠", layout="wide")
st.title("Fashion Product Image Classification")
st.caption("A Convolutional Neural Network (CNN) trained on Fashion-MNIST — a deep learning project, not a Generative AI application.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Open the VS Code terminal and run `python train.py` first.")
    st.stop()

@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def get_test_data():
    return load_fashion_mnist()[1]

model = get_model()
(x_test, y_test) = get_test_data()

st.write(
    "Select a test-set image using the slider. Click **Predict image** to see "
    "the CNN prediction, confidence, and probability for each class."
)
index = st.slider("Test image index", min_value=0, max_value=len(x_test) - 1, value=0)
image = x_test[index]
true_label = int(y_test[index])

left, right = st.columns([1, 2])
with left:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"Test image #{index}")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"**Actual class:** {CLASS_NAMES[true_label]}")

with right:
    if st.button("Predict image", type="primary"):
        probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])
        if predicted_label == true_label:
            st.success(f"Prediction: {CLASS_NAMES[predicted_label]} ({confidence:.1%} confidence)")
        else:
            st.warning(f"Prediction: {CLASS_NAMES[predicted_label]} ({confidence:.1%} confidence)")
        st.write(f"**Actual class:** {CLASS_NAMES[true_label]}")
        st.subheader("Class probabilities")
        order = np.argsort(probabilities)[::-1]
        for class_id in order:
            st.write(f"{CLASS_NAMES[class_id]} — {probabilities[class_id]:.1%}")
            st.progress(float(probabilities[class_id]))

st.divider()
st.subheader("Model information")
st.markdown("""
- **Model:** Convolutional Neural Network (CNN)
- **Input:** 28 × 28 grayscale image
- **Output:** One of 10 Fashion-MNIST clothing classes
- **Core concepts:** convolution, max pooling, ReLU, Softmax, dropout, Adam, cross-entropy, validation, and test evaluation
""")
st.info("This demonstration uses Fashion-MNIST test images. It is not designed to recognize arbitrary photographs of clothing.")
