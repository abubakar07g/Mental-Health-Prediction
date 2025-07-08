import streamlit as st
import pickle
from textblob import TextBlob
from bs4 import BeautifulSoup

# Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()  # Remove HTML
    return text.lower().strip()

# Predict function
def predict_depression(text):
    cleaned = clean_text(text)
    polarity = TextBlob(cleaned).sentiment.polarity

    if polarity > 0.3:  # lower threshold
        return f"Not Depressed 🙂"
    
    # ML-based classification
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]
    return f"{'Depressed 😞' if prediction == 1 else 'Not Depressed 🙂'}"


# Streamlit UI
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠")
st.title("🧠 Predict Mental Health from User's Text")

st.markdown("Enter a tweet or message to check if it's **Depressed** or **Not Depressed**.")

user_input = st.text_area("✍️ Enter Text Here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = predict_depression(user_input)
        st.success(f"🧾 Prediction: {result}")
