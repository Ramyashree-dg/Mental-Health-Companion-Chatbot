import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# -------------------------------------------------
# OPTIMIZATION FOR DEPLOYMENT
# -------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Mental Health Companion",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM DARK UI
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

.user-bubble {
    background-color: #2E86C1;
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 10px;
    animation: fadeIn 0.4s ease-in;
}

.bot-bubble {
    background-color: #1C2833;
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 10px;
    animation: fadeIn 0.4s ease-in;
}

.stButton>button {
    background-color: #E74C3C;
    color: white;
    border-radius: 12px;
    padding: 8px 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #C0392B;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER + DISCLAIMER
# -------------------------------------------------
st.markdown("<h1 style='text-align:center; color:var(--text-color);'>🧠 Mental Health Companion Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:var(--text-color);'>Your Safe Space to Share and Reflect 💙</h4>", unsafe_allow_html=True)

st.error("⚠️ Disclaimer: This chatbot provides emotional support but is NOT a substitute for professional medical advice.")

st.markdown("---")

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_emotion_model():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )

with st.spinner("Loading AI model... Please wait ⏳"):
    emotion_pipeline = load_emotion_model()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood_history" not in st.session_state:
    st.session_state.mood_history = []

if "used_tips" not in st.session_state:
    st.session_state.used_tips = []

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------
def detect_emotion(text):
    result = emotion_pipeline(text)[0]
    return result["label"]

def check_crisis(text):
    keywords = ["suicide", "kill myself", "die", "end my life", "hurt myself"]
    return any(word in text.lower() for word in keywords)

def generate_response(emotion):
    responses = {
        "sadness": "I'm really sorry you're feeling sad. Your feelings are valid, and it's okay to take time to heal.",
        "anger": "It sounds like you're feeling frustrated. Let’s pause and take a deep breath together.",
        "fear": "Anxiety can feel overwhelming. You’re not alone — let's slow down.",
        "joy": "That’s wonderful to hear! I’m glad you're feeling joyful.",
        "love": "That’s beautiful. Human connection is powerful.",
        "surprise": "That sounds unexpected! Want to share more about it?"
    }
    return responses.get(emotion, "I'm here to listen.")

def get_tip():
    tips = [
        "Try the 4-7-8 breathing technique.",
        "Take a 5-minute mindful walk.",
        "Stretch your shoulders gently.",
        "Write down three things you're grateful for.",
        "Drink water and breathe slowly."
    ]

    # Reset when all tips used
    if len(st.session_state.used_tips) == len(tips):
        st.session_state.used_tips = []

    remaining = [tip for tip in tips if tip not in st.session_state.used_tips]
    tip = random.choice(remaining)
    st.session_state.used_tips.append(tip)
    return tip

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
chat_col, mood_col = st.columns([3, 1])

# ---------------- CHAT SECTION ----------------
with chat_col:

    # Display messages
    for role, message in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{message}</div>", unsafe_allow_html=True)

    user_input = st.chat_input("How are you feeling today?")

    if user_input:
        st.session_state.messages.append(("user", user_input))

        with st.spinner("AI is thinking..."):
            time.sleep(2)

            if check_crisis(user_input):
                response = "🚨 Please contact emergency services or a trusted person immediately. You are not alone."
            else:
                emotion = detect_emotion(user_input)
                st.session_state.mood_history.append(emotion)
                response = generate_response(emotion)
                response += f"\n\n💡 Relaxation Tip: {get_tip()}"

        st.session_state.messages.append(("bot", response))
        st.rerun()

    # Reset Button
    if st.session_state.messages:
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = []
            st.session_state.mood_history = []
            st.session_state.used_tips = []
            st.rerun()

# ---------------- MOOD PANEL ----------------
with mood_col:
    if st.session_state.mood_history:
        st.markdown("""
        <h3 style='color: var(--text-color);'>
        📊 Mood Trends
        </h3>
        """, unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.mood_history, columns=["Emotion"])
        counts = df["Emotion"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)

        st.pyplot(fig)
