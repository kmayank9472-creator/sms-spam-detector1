import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("spam_detector_pipeline.pkl")

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("SMS Spam Detector")
st.write("SMS likhein aur check karein ki woh spam hai ya nahi.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    message = st.text_area("SMS message", placeholder="Yahan SMS paste karein…")

    if st.button("Check SMS"):
        if not message.strip():
            st.warning("Pehle SMS message likhein.")
        else:
            prediction = model.predict([message])[0]
            label = str(prediction).strip().lower()

            if label in {"1", "spam", "true"}:
                st.error("🚨 Spam")
            elif label in {"0", "ham", "not spam", "not_spam", "false"}:
                st.success("✅ Not Spam")
            else:
                st.info(f"Model prediction: {prediction}")

except FileNotFoundError:
    st.error("spam_detector_pipeline.pkl file app.py ke saath upload karein.")
except Exception as error:
    st.error(f"Error: {error}")
