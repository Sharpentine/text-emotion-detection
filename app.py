import streamlit as st
import joblib

# Load model and preprocessors
svm_model = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# Label mapping
label_to_emotion = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

st.title("Emotion Classification with SVM")
st.write("Enter a sentence to predict its emotion:")

user_input = st.text_area("Text", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        X_tfidf = tfidf_vectorizer.transform([user_input])
        X_scaled = scaler.transform(X_tfidf)
        # Predict
        pred = svm_model.predict(X_scaled)[0]
        emotion = label_to_emotion.get(pred, "Unknown")
        st.success(f"**Predicted Emotion:** {emotion}")