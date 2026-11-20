
import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SMS Shield AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172b46);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    background: linear-gradient(120deg, #2563eb, #7c3aed);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    color: white !important;
    font-size: 38px;
}
.hero p {
    color: #e0e7ff !important;
}
.result {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}
.spam {
    background: #451a1a;
    border: 1px solid #ef4444;
    color: #fecaca;
}
.ham {
    background: #123b2a;
    border: 1px solid #22c55e;
    color: #bbf7d0;
}
div[data-testid="stMetric"] {
    background: #20344e;
    padding: 15px;
    border-radius: 12px;
}
.stButton button {
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    path = "spam_detector_pipeline.pkl"

    if not os.path.exists(path):
        return None

    return joblib.load(path)

model = load_model()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "message" not in st.session_state:
    st.session_state.message = ""

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛡️ SMS SHIELD AI</h1>
    <p>Smart SMS Spam Detection Using Machine Learning</p>
    <p>TF-IDF + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.write("**SMS Shield AI**")
    st.caption("Machine Learning Project")

    st.divider()

    st.write("📌 Features")
    st.write("✅ Spam Detection")
    st.write("✅ Not Spam Detection")
    st.write("✅ Confidence Analysis")
    st.write("✅ Prediction History")

    st.divider()

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.5, 1])

with left:
    st.subheader("📩 Message Analyzer")

    examples = {
        "Select an example": "",
        "Spam Example": "Congratulations! You won a free prize. Call now to claim.",
        "Not Spam Example": "Hello, I will meet you tomorrow at college.",
        "Bank Example": "Your account statement is ready. Please check your bank app."
    }

    selected = st.selectbox(
        "Try a sample SMS",
        list(examples.keys())
    )

    if selected != "Select an example":
        st.session_state.message = examples[selected]

    sms = st.text_area(
        "Enter your SMS message",
        key="message",
        height=160,
        placeholder="Type or paste your SMS here..."
    )

    st.caption(
        f"Characters: {len(sms)} | Words: {len(sms.split())}"
    )

    analyze = st.button(
        "🚀 ANALYZE MESSAGE",
        use_container_width=True
    )

with right:
    st.subheader("📊 Message Overview")

    st.metric("Characters", len(sms))
    st.metric("Words", len(sms.split()))
    st.metric("Model Status", "Ready" if model else "Not Found")

    st.info(
        "Our AI model checks message patterns "
        "and predicts Spam or Not Spam."
    )

# ---------------- PREDICTION ----------------
if analyze:

    if model is None:
        st.error(
            "Model file not found! Keep "
            "spam_detector_pipeline.pkl in the same folder."
        )

    elif not sms.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            prediction = int(model.predict([sms])[0])

            spam_probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([sms])[0]
                classes = list(model.classes_)

                if 1 in classes:
                    spam_probability = float(
                        probabilities[classes.index(1)]
                    )

            st.divider()
            st.subheader("🔍 AI Analysis Result")

            if prediction == 1:
                st.markdown("""
                <div class="result spam">
                    🚨 SPAM MESSAGE DETECTED
                </div>
                """, unsafe_allow_html=True)

                st.warning(
                    "This message has been classified as Spam. "
                    "Avoid clicking suspicious links."
                )

            else:
                st.markdown("""
                <div class="result ham">
                    ✅ THIS IS NOT SPAM
                </div>
                """, unsafe_allow_html=True)

                st.success(
                    "This message has been classified as Not Spam."
                )

            # Probability Analysis
            if spam_probability is not None:
                st.subheader("📈 Spam Probability")

                st.progress(spam_probability)

                st.write(
                    f"Spam Probability: {spam_probability * 100:.2f}%"
                )

                st.write(
                    f"Not Spam Probability: "
                    f"{(1 - spam_probability) * 100:.2f}%"
                )

            # Save History
            st.session_state.history.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Message": sms[:55],
                "Result": "Spam" if prediction == 1 else "Not Spam"
            })

            st.session_state.history = (
                st.session_state.history[:10]
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ---------------- HISTORY ----------------
st.divider()
st.subheader("🕘 Recent Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No messages analyzed yet.")

# ---------------- FOOTER ----------------
st.divider()

st.markdown("""
<center>
    <h3>🛡️ SMS Shield AI</h3>
    <p>Developed using Python, Streamlit and Machine Learning</p>
    <p>Algorithm: TF-IDF + Logistic Regression</p>
</center>
""", unsafe_allow_html=True)
