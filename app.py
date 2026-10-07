import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import CountVectorizer

from nlp_utils import clean_text

st.set_page_config(page_title="MESCOE College Portal & AI Chatbot", page_icon="🎓", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
        div[data-testid="stPopover"] {
            position: fixed !important; bottom: 30px !important; right: 30px !important;
            z-index: 999999 !important;
        }
        div[data-testid="stPopover"] > button {
            background-color: #1e3a8a !important; color: white !important;
            border-radius: 50px !important; padding: 12px 24px !important;
            font-size: 16px !important; font-weight: bold !important;
            border: 2px solid white !important; box-shadow: 0px 4px 15px rgba(0,0,0,0.3) !important;
        }
        div[data-testid="stPopover"] > button:hover {
            background-color: #2563eb !important; transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ load artifacts
@st.cache_resource
def load_assets():
    model = joblib.load("mescoe_chatbot_pipeline.pkl")
    dataset = pd.read_csv("mescoe_dataset.csv")
    metrics_df = pd.read_csv("model_metrics.csv")
    with open("model_config.json") as f:
        config = json.load(f)
    return model, dataset, metrics_df, config


try:
    model, dataset, metrics_df, config = load_assets()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Run these first:  `python build_dataset.py`  then  `python train_model.py`")
    st.stop()

intent_response_map = dict(zip(dataset["intent"], dataset["response"]))
FALLBACK = ("I'm not sure I understood that. Could you rephrase? "
            "You can also contact info@mescoepune.org for details.")
WELCOME = "Hello! Welcome to MES Wadia College of Engineering. How can I assist you today?"


def predict(question: str, threshold: float) -> dict:
    """Classify a question and return the intent, confidence, answer and top-3 intents."""
    probs = model.predict_proba([clean_text(question)])[0]
    order = probs.argsort()[::-1][:3]
    top3 = [(str(model.classes_[i]), float(probs[i])) for i in order]
    intent, conf = top3[0]
    if conf < threshold:
        return {"intent": "low confidence", "confidence": conf, "answer": FALLBACK, "top3": top3}
    return {"intent": intent, "confidence": conf, "answer": intent_response_map[intent], "top3": top3}


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎓 MESCOE AI Assistant")
st.sidebar.caption("TE Machine Learning Mini Project")
threshold = st.sidebar.slider(
    "Confidence threshold", 0.05, 0.90, float(config.get("confidence_threshold", 0.3)), 0.05,
    help="If the model's confidence is below this value the bot asks you to rephrase.",
)
st.sidebar.markdown("**Try asking:**")
st.sidebar.markdown(
    "- How many seats are in Computer Engineering?\n"
    "- Is hostel available for girls?\n"
    "- Which companies visit for placements?\n"
    "- What documents are required for admission?\n"
    "- Who is the principal?\n"
    "- Do you offer M.E. courses?"
)
st.sidebar.info(f"Model in use: **{config['best_model']}**\n\n"
                f"Trained on {config['n_samples']} questions across {config['n_intents']} intents.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏛️ Website & AI Chatbot", "🔍 Intent Predictor", "📈 EDA", "📊 Model Performance",
    "📁 Dataset", "👥 Team & About",
])

# ------------------------------------------------------------------ TAB 1: website + floating chatbot
with tab1:
    website_html = r"""
    <!DOCTYPE html><html><head><style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }
        header { background: #1e3a8a; color: white; padding: 18px 40px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { margin: 0; font-size: 22px; }
        nav a { color: white; margin-left: 20px; text-decoration: none; font-weight: 500; font-size: 15px; }
        .hero { background-color: #1e3a8a; background: linear-gradient(135deg, #1e3a8a, #2563eb); color: white; padding: 60px 20px; text-align: center; }
        .hero h2 { font-size: 32px; margin-bottom: 10px; }
        .hero p { font-size: 18px; opacity: 0.9; }
        .cards { display: flex; gap: 20px; padding: 40px 20px; justify-content: center; flex-wrap: wrap; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 200px; text-align: center; border-top: 4px solid #1e3a8a; }
        .card h3 { color: #1e3a8a; margin-top: 0; font-size: 18px; }
        .card p { font-size: 14px; color: #475569; margin-top: 10px; line-height: 1.5; }
        footer { background: #0f172a; color: #94a3b8; text-align: center; padding: 20px; font-size: 14px; margin-top: 20px; }
    </style></head><body>
    <header>
        <h1>Modern Education Society's Wadia College of Engineering</h1>
        <nav><a href="#">Home</a><a href="#">Admissions</a><a href="#">Departments</a><a href="#">Placements</a></nav>
    </header>
    <div class="hero"><h2>Welcome to MESCOE Pune</h2>
        <p>NAAC 'A++' Grade Accredited Institute | SPPU Affiliated</p></div>
    <div class="cards">
        <div class="card"><h3>Computer Engg</h3><p>UG Intake: 300<br>PG Intake: 12</p></div>
        <div class="card"><h3>E & TC Engg</h3><p>UG Intake: 120<br>PG Intake: 12</p></div>
        <div class="card"><h3>Mechanical Engg</h3><p>UG Intake: 60<br>PG Intake: 12</p></div>
        <div class="card"><h3>Automation & Robotics</h3><p>UG Intake: 60</p></div>
    </div>
    <footer><p>© Modern Education Society's Wadia College of Engineering, Pune. All Rights Reserved.</p></footer>
    </body></html>
    """
    components.html(website_html, height=520, scrolling=True)

    with st.popover("💬 Chat with AI Assistant"):
        st.subheader("🤖 MESCOE AI Chatbot")
        st.caption("Ask about courses, admissions, fees, hostel, placements and more.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME, "meta": ""}]
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME, "meta": ""}]

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])

        if user_input := st.chat_input("Ask a question..."):
            result = predict(user_input, threshold)
            meta = f"Detected intent: {result['intent']}  |  confidence: {result['confidence']:.0%}"
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                st.write(result["answer"])
                st.caption(meta)
            st.session_state.chat_messages.append({"role": "user", "content": user_input, "meta": ""})
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": result["answer"], "meta": meta})

# ------------------------------------------------------------------ TAB 2: intent predictor (input form)
with tab2:
    st.header("🔍 Intent Predictor")
    st.write("Type any question. The model predicts its **intent** and shows how confident it is.")
    with st.form("predict_form"):
        question = st.text_input("Your question", placeholder="e.g. How many seats are there in E&TC?")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            res = predict(question, threshold)
            c1, c2 = st.columns(2)
            c1.metric("Predicted intent", res["intent"])
            c2.metric("Confidence", f"{res['confidence']:.0%}")
            st.success(res["answer"])
            st.subheader("Top 3 candidate intents")
            top_df = pd.DataFrame(res["top3"], columns=["Intent", "Probability"]).set_index("Intent")
            st.bar_chart(top_df)

# ------------------------------------------------------------------ TAB 3: EDA
with tab3:
    st.header("📈 Exploratory Data Analysis")
    eda = dataset.copy()
    eda["clean"] = eda["text"].map(clean_text)
    eda["word_count"] = eda["clean"].str.split().str.len()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total questions", len(eda))
    c2.metric("Intents", eda["intent"].nunique())
    c3.metric("Avg words / question", f"{eda['word_count'].mean():.1f}")
    c4.metric("Duplicates", int(eda["clean"].duplicated().sum()))

    st.subheader("Samples per intent")
    counts = eda["intent"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(counts.index, counts.values, color="#1e3a8a")
    ax.set_xlabel("Number of questions")
    st.pyplot(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Question length (words)")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(eda["word_count"], bins=range(1, int(eda["word_count"].max()) + 2),
                color="#2563eb", edgecolor="white")
        ax.set_xlabel("Words per question")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with right:
        st.subheader("Most frequent words")
        cv = CountVectorizer(stop_words="english")
        bow = cv.fit_transform(eda["clean"])
        freq = pd.Series(bow.sum(axis=0).A1, index=cv.get_feature_names_out()).nlargest(15)[::-1]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(freq.index, freq.values, color="#1e3a8a")
        ax.set_xlabel("Frequency")
        st.pyplot(fig)

    st.subheader("Statistical summary of question length")
    st.dataframe(eda["word_count"].describe().to_frame("word_count"))

# ------------------------------------------------------------------ TAB 4: model performance
with tab4:
    st.header("📊 Model Performance")
    st.write(f"**Selected model:** {config['best_model']} (highest cross-validated F1-score)")
    st.caption("Accuracy / Precision / Recall / F1 are measured on a held-out 20% test split. "
               "'Challenge Accuracy' uses hand-written questions with wording never seen in training.")
    st.dataframe(metrics_df)

    plot_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "Challenge Accuracy"]
                 if c in metrics_df.columns]
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_df.set_index("Model")[plot_cols].plot(kind="bar", ax=ax, rot=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)

    if os.path.exists("confusion_matrix.png"):
        st.subheader("Confusion matrix (held-out test set)")
        st.image("confusion_matrix.png")
    if os.path.exists("classification_report.txt"):
        with st.expander("Per-intent classification report"):
            with open("classification_report.txt") as f:
                st.text(f.read())

# ------------------------------------------------------------------ TAB 5: dataset preview
with tab5:
    st.header("📁 Dataset Preview")
    st.caption("Custom dataset created by the team: student questions labelled by intent, "
               "with answers based on public MESCOE information.")
    choice = st.selectbox("Filter by intent", ["All"] + sorted(dataset["intent"].unique()))
    view = dataset if choice == "All" else dataset[dataset["intent"] == choice]
    st.write(f"Showing {len(view)} rows")
    st.dataframe(view)

# ------------------------------------------------------------------ TAB 6: team & about
with tab6:
    st.header("👥 Team & About")
    st.subheader("Project")
    st.write("**MESCOE College Portal with an ML-powered AI Assistant** - an intent-classification "
             "chatbot that answers student queries about the college.")
    st.markdown(
        "- **Problem type:** Multi-class text classification (27 intents)\n"
        "- **Features:** TF-IDF on words (1-2 grams) + characters (2-5 grams)\n"
        "- **Models compared:** Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest\n"
        "- **Tools:** Python, Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit"
    )

    st.subheader("Team members")

    st.table(pd.DataFrame({
        "Name": ["Paras Dalvi", "Pranav Kulkarni"],
        "Role": ["ML Developer", "Web/App Developer"],
        "Contribution": [
            "ML model, dataset and training",
            "Streamlit app & documentation"
        ]
    }))
    st.write("**Guide:** Prof.Aparna Kulkarni |  **Department:** E&TC Engineering, MESCOE Pune")
    st.subheader("References")
    st.markdown(
        "- MESCOE official website - https://mescoepune.org\n"
        "- Scikit-learn documentation - https://scikit-learn.org\n"
        "- Streamlit documentation - https://docs.streamlit.io"
    )
