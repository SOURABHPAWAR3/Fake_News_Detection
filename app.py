# app.py

import streamlit as st
import joblib
import os
from dotenv import load_dotenv
from preprocess import clean_text
from newsapi_utils import fetch_top_headlines

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Ensure this exists in your .env file

# Load model and vectorizer
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Set page config
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection")

st.write(
    "Check if the news is **REAL** or **FAKE**. You can either:\n"
    "🔹 Click **Fetch & Analyze Live News** to test with real-time headlines, or\n"
    "🔹 Paste any custom news headline below to analyze it manually."
)

# Section 1: Live News Headlines
st.subheader("📰 Try with Live Headlines")

# Optional: Dropdown to select category
selected_category = st.selectbox("📂 Choose News Category:", [
    'technology', 'business', 'sports', 'health', 'entertainment', 'science'
])

if st.button("🔁 Fetch & Analyze Live News"):
    if not NEWSAPI_KEY:
        st.warning("⚠️ NewsAPI key not found. Please set it in your `.env` file.")
    else:
        headlines = fetch_top_headlines(NEWSAPI_KEY, category=selected_category, limit=5)

        if not headlines:
            st.error("❌ No headlines returned.")
        elif len(headlines) == 1 and headlines[0].startswith(("❌", "🔐", "⏱", "⚠️", "📭")):
            st.warning(f"{headlines[0]}")
        else:
            for headline in headlines:
                st.markdown(f"**🔸 {headline}**")

                cleaned = clean_text(headline)
                vectorized = vectorizer.transform([cleaned])
                prediction = model.predict(vectorized)[0]
                label = "REAL" if prediction == 1 else "FAKE"
                confidence = model.predict_proba(vectorized).max()

                if label == "REAL":
                    st.success(f"✅ Prediction: REAL | Confidence: {confidence:.2f}")
                else:
                    st.error(f"❌ Prediction: FAKE | Confidence: {confidence:.2f}")

# Section 2: Manual Input
st.subheader("🔍 Analyze Your News Headline")

input_text = st.text_area("✍️ Enter a news headline or short article text:")

if st.button("🚀 Detect"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        confidence = model.predict_proba(vectorized).max()

        if label == "REAL":
            st.success(f"✅ This news is likely **REAL**.")
        else:
            st.error(f"❌ This news is likely **FAKE**.")

        st.markdown(f"🧠 **Model Confidence:** `{confidence:.2f}`")
