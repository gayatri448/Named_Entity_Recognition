import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle, json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ------------------ Load Model ------------------
model = tf.keras.models.load_model(
    "model/ner_model.h5",
    custom_objects={
        "masked_loss": lambda y_true, y_pred: 0,
        "masked_accuracy": lambda y_true, y_pred: 0
    }
)

# ------------------ Load Vectorizer ------------------
with open("model/sentence_vectorizer.pkl", "rb") as f:
    saved = pickle.load(f)

sentence_vectorizer = tf.keras.layers.TextVectorization.from_config(saved["config"])
sentence_vectorizer.set_weights(saved["weights"])

# ------------------ Load Tag Map ------------------
with open("model/tag_map.json", "r") as f:
    tag_map = json.load(f)

id2tag = {v: k for k, v in tag_map.items()}

# ------------------ Prediction Function ------------------
def predict_ner(sentence):
    # Vectorize sentence
    sent_vec = sentence_vectorizer(tf.constant([sentence]))
    
    # Get predictions
    preds = model(sent_vec).numpy()
    
    # Get predicted tag ids
    pred_ids = np.argmax(preds, axis=-1)[0]

    # Map ids to words
    words = sentence.split()
    tags = [id2tag[i] for i in pred_ids[:len(words)]]
    
    return list(zip(words, tags))

# ------------------ Streamlit UI ------------------
st.title("🌟 Named Entity Recognition App")
st.write("Enter a sentence and extract entities using LSTM model.")

text = st.text_area("Enter text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text!")
    else:
        result = predict_ner(text)
        st.subheader("Entities Found:")
        for word, tag in result:
            if tag != "O":
                st.markdown(f"**{word} → {tag}**")
