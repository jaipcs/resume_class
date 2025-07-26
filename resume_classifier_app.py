import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from docx import Document
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model artifacts
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# Main extraction function (DOCX only)
def extract_text_from_resume(file):
    if file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""

# Send email function
def send_email(to_email, candidate_name, predicted_role):
    sender_email = "your_email@gmail.com"   # Replace with your Gmail
    password = "your_app_password"          # Gmail App Password

    subject = f"Interview Invitation for {predicted_role}"
    body = f"""
    Dear {candidate_name},

    We reviewed your resume and found your skills suitable for the {predicted_role} role.
    We would like to invite you for an interview.

    Regards,
    HR Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True
    except Exception:
        return False

# Feature importance
def show_feature_importance(model, tfidf, top_n=20):
    if hasattr(model, "feature_importances_"):
        feature_names = tfidf.get_feature_names_out()
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        top_features = pd.DataFrame({
            "Feature": [feature_names[i] for i in indices],
            "Importance": [importances[i] for i in indices]
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(top_features.set_index("Feature"))
    else:
        st.info("Feature importance not available for this model type.")

# Streamlit UI

import streamlit as st

st.title("Resume Classifier App (DOCX Only)")
st.markdown("Upload your resume in DOCX format to predict job role and send interview invitation.")

uploaded_file = st.file_uploader("Upload Resume", type=["docx"])
candidate_email = st.text_input("Candidate Email")
candidate_name = st.text_input("Candidate Name")

if uploaded_file is not None:
    resume_text = extract_text_from_resume(uploaded_file)
    cleaned_text = clean_text(resume_text)

    # Prediction
    X_input = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(X_input)[0]
    predicted_role = le.inverse_transform([prediction])[0]

    st.subheader(f"Predicted Role: {predicted_role}")

    # Feature importance chart
    st.subheader("Top Keywords Influencing Prediction")
    show_feature_importance(model, tfidf)

    # Send email option
    if candidate_email and st.button("Send Interview Invitation"):
        masked_email = candidate_email[:3] + "***@" + candidate_email.split("@")[1]
        success = send_email(candidate_email, candidate_name or "Candidate", predicted_role)
        if success:
            st.success(f"Interview email sent to {masked_email}")
        else:
            st.error("Failed to send email. Check SMTP/App Password settings.")