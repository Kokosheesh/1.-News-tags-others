import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# TRAINING DATA (mock headlines)
# -------------------------------
data = {
    'headline': [
        "Markets rally as inflation eases",
        "TCS beats earnings estimates in Q4 results",
        "Sensex crashes 1000 points amid global tensions",
        "Inflation hits 7-month high, markets dip",
        "RBI holds interest rates steady",
        "Budget 2025 to focus on infra spending",
        "Cricket World Cup 2023 schedule announced",
        "India's population hits record high",
    ],
    'sentiment': [
        "bullish", "bullish",
        "bearish", "bearish",
        "neutral", "neutral",
        "not applicable", "not applicable"
    ]
}
df = pd.DataFrame(data)

# -------------------------------
# ML PIPELINE
# -------------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
model.fit(df['headline'], df['sentiment'])

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Headline Classifier", page_icon="🧠")

st.markdown("<h1 style='text-align: center;'>📰 Headline Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Classify finance headlines as <strong>Bullish, Bearish, Neutral</strong> or <strong>Not Applicable</strong></p>", unsafe_allow_html=True)
st.markdown("---")

headline_input = st.text_area("🖊️ Enter a news headline", height=100, placeholder="e.g. RBI hikes repo rate by 50 bps amid inflation fears")

if st.button("🔍 Classify"):
    if headline_input.strip() == "":
        st.warning("Please enter a headline first.")
    else:
        prediction = model.predict([headline_input])[0]
        
        st.markdown("### 📊 Prediction:")
        st.success(f"**{prediction.upper()}**")

        with st.expander("ℹ️ What each label means"):
            st.markdown("""
            - **Bullish** 🟢: Suggests positive market movement
            - **Bearish** 🔴: Indicates possible downturn or risk
            - **Neutral** ⚪: News that doesn't signal strong movement
            - **Not Applicable** 🚫: Irrelevant to financial markets
            """)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
"Add Streamlit app"
