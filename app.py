import streamlit as st
from transformers import pipeline

# -----------------------------
# 1. Model + docs loading
# -----------------------------

# Use a tiny model so it fits on Streamlit Cloud
MODEL_NAME = "google/gemma-2b-it"

import os

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

@st.cache_resource
def load_model():
    text_gen = pipeline(
        "text-generation",
        model=MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
        max_new_tokens=256,
        temperature=0.4,
    )
    return text_gen

@st.cache_resource
def load_docs():
    def read_file(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    docs = {
        "ai_basics": read_file("ai.txt"),
        "ml": read_file("ml.txt"),
        "gen_ai": read_file("genai.txt"),
    }
    return docs


text_gen = load_model()
docs = load_docs()

# -----------------------------
# 2. RAG helper functions
# -----------------------------

def get_relevant_docs(question: str, top_n: int = 2):
    words = [w for w in question.lower().split() if len(w) > 3]
    scores = []
    for name, content in docs.items():
        content_lower = content.lower()
        score = sum(content_lower.count(w) for w in words)
        scores.append((score, name))

    scores.sort(reverse=True)
    return [name for s, name in scores if s > 0][:top_n]


def ask_ai_learning_assistant(question: str, level: str = "beginner"):
    selected = get_relevant_docs(question, top_n=2)
    context = "\n\n".join(
        [f"[{name}]\n{docs[name]}" for name in selected]
    ) if selected else "No relevant content found."

    level_lower = level.lower()

    if "begin" in level_lower:
        tone = (
            "Explain like I'm a beginner who is new to AI and machine learning. "
            "Use simple language, short sentences, and concrete examples. "
            "Avoid formulas unless absolutely necessary. Include a short recap at the end."
        )
    elif "inter" in level_lower:
        tone = (
            "Explain for an intermediate learner who knows basic AI/ML terms "
            "(like model, training, dataset, overfitting) but wants deeper intuition. "
            "You can use some math notation and technical jargon but always explain it."
        )
    else:
        tone = (
            "Explain for an advanced learner (engineer / data scientist). "
            "You may use equations, technical jargon, and go into implementation details. "
            "Focus on precision and depth, and mention trade-offs and design choices."
        )

    prompt = f"""
You are an AI Learning Assistant for topics in Artificial Intelligence, Machine Learning, and Generative AI.

Your goals:
1. Use ONLY the context below to answer. Do not invent facts.
2. If the answer is not present in the context, reply exactly:
   "Not available in current documents."
3. Act like a patient tutor: break down concepts, give intuition, and suggest what to learn next.

Tone & audience:
{tone}

Context:
{context}

---
Question from learner:
{question}

Now write the best possible answer as the tutor.
Include:
- a clear explanation,
- an example or analogy if appropriate,
- and 2–3 suggested follow-up topics to study.
Answer:
"""

    result = text_gen(prompt)[0]["generated_text"]

    # tiny-gpt2 will echo the prompt; keep it simple and just return the tail
    if prompt in result:
        answer = result.split(prompt, 1)[1].strip()
    else:
        answer = result.strip()

    return answer, selected


# -----------------------------
# 3. Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 AI Learning Assistant")
    st.write(
        "Ask questions about **AI, Machine Learning, and Generative AI**. "
        "Answers are generated from your knowledge files (`ai.txt`, `ml.txt`, `genai.txt`)."
    )

    with st.sidebar:
        st.header("Settings")
        level = st.selectbox(
            "Difficulty level",
            ["beginner", "intermediate", "advanced"],
            index=0,
        )
        show_sources = st.checkbox("Show source docs used", value=True)

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the difference between supervised and unsupervised learning?",
        height=100,
    )

    if st.button("Get answer", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            answer, used_docs = ask_ai_learning_assistant(question, level=level)

        st.subheader("Answer")
        st.markdown(answer)

        if show_sources:
            st.markdown("---")
            st.caption(
                "📚 Sources used: " + (", ".join(used_docs) if used_docs else "None")
            )


if __name__ == "__main__":
    main()

