
import streamlit as st
import joblib
import re
import html
import os

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "toxic_model.joblib"))
    tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.joblib"))
    labels = joblib.load(os.path.join(BASE_DIR, "labels.joblib"))
    return model, tfidf, labels

def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("🛡️ Toxic Comment Detection System")
st.write("Analyze online comments and identify potentially harmful content.")

st.info(
    "This tool supports moderation review. Predictions can be wrong, "
    "so review flagged comments before taking action."
)

try:
    model, tfidf, labels = load_model()
except Exception as e:
    st.error("Unable to load the saved model files.")
    st.code(str(e))
    st.stop()

comment = st.text_area(
    "Enter a comment",
    placeholder="Type or paste a comment here...",
    height=150
)

if st.button("Analyze Comment", use_container_width=True):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        features = tfidf.transform([clean_text(comment)])
        prediction = model.predict(features)[0]

        predicted_labels = [
            label for label, value in zip(labels, prediction)
            if value == 1
        ]

        st.subheader("Analysis Result")

        if predicted_labels:
            st.error("Potentially toxic content detected")
            for label in predicted_labels:
                st.write("⚠️", label.replace("_", " ").title())
        else:
            st.success("No toxicity labels predicted")
            st.write(
                "The model did not flag this comment. "
                "This does not guarantee the comment is harmless."
            )

        with st.expander("View all label predictions"):
            for label, value in zip(labels, prediction):
                status = "Flagged" if value else "Not flagged"
                st.write(f"{label.replace('_', ' ').title()}: {status}")

st.divider()
st.caption("NLP | TF-IDF | One-vs-Rest Logistic Regression")
