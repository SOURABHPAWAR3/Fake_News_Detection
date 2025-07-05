# 📰 Fake News Detection using Machine Learning

Detect whether a news headline is **REAL** or **FAKE** using Natural Language Processing and Machine Learning. The app supports both **manual text input** and **live news headlines** fetched via NewsAPI.

---

## 📌 Features

- 🔎 Detect fake or real news from any text input
- 🔁 Fetch and analyze **live news headlines** from NewsAPI
- ✅ Confidence score shown with each prediction
- 📂 Dropdown to choose news category: technology, sports, business, etc.
- 💡 Clean, interactive UI built with Streamlit

---

## 🧠 Tech Stack

| Layer         | Technology                  |
|---------------|------------------------------|
| Language      | Python 3.x                   |
| ML Model      | Logistic Regression          |
| Vectorization | TF-IDF (NLP Preprocessing)   |
| Libraries     | Scikit-learn, Pandas, NLTK   |
| Frontend      | Streamlit                    |
| API           | NewsAPI.org                  |
| Deployment    | Streamlit Cloud              |

---

## 📁 Project Structure
FakeNewsDetection/
├── app.py # Streamlit main app
├── newsapi_utils.py # Fetch headlines via NewsAPI
├── preprocess.py # Clean and process text
├── models/
│ ├── logistic_model.pkl
│ └── tfidf_vectorizer.pkl
├── data/
│ ├── True.csv
│ └── Fake.csv
├── .env # [Not pushed] contains NewsAPI key
├── .gitignore
├── requirements.txt
└── README.md




