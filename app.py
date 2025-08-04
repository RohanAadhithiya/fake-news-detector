import streamlit as st
import pickle
import numpy as np
import os



# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Title
st.title("📰 Fake News Detector")
st.markdown("Enter a news article or paragraph below to check if it's **Real** or **Fake**.")

# Text input box
user_input = st.text_area("Paste your news content here:")

# Button to trigger prediction
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.error("❌ This news is *Fake*.")
        else:
            st.success("✅ This news is *Real*.")
