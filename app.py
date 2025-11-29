import streamlit as st

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

st.title("🎓 AI Learning Assistant")
st.write(
    "Ask questions about **AI, Machine Learning, and Generative AI**.\n\n"
    "Right now this is a UI-only demo (no real model yet)."
)

with st.sidebar:
    st.header("Settings")
    level = st.selectbox(
        "Difficulty level",
        ["beginner", "intermediate", "advanced"],
        index=0
    )

question = st.text_area(
    "Your question",
    placeholder="e.g. What is the difference between supervised and unsupervised learning?",
    height=100,
)

if st.button("Get answer", type="primary") and question.strip():
    st.subheader("Answer")
    st.markdown(
        f"""
        _(Demo mode)_  
        You asked: **{question}**  
        Level selected: **{level}**  

        This is where the AI/RAG answer from your Databricks knowledge base would appear.
        """
    )
