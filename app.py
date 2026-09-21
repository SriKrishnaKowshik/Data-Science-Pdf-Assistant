
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(page_title="Data Science PDF Assistant", page_icon="📚")

st.title("📚 Data Science PDF Assistant")
st.write("Ask questions from your uploaded Machine Learning PDFs.")

# -----------------------------
# Load models (only once)
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("pdf_collection")

model = load_embedding_model()
collection = load_chroma()

# -----------------------------
# User input
# -----------------------------
question = st.text_input("Ask your question")

if st.button("Generate Answer") and question:

    with st.spinner("Searching documents..."):

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = ""

        for doc, meta in zip(documents, metadatas):
            context += f"""
Source: {meta["source"]}
Page: {meta["page"]}

{doc}

-------------------
"""

    prompt = f"""
You are an expert Machine Learning tutor.

Answer ONLY from the provided context.

If the answer is missing, reply:
I couldn't find it in the uploaded PDFs.

Context:
{context}

Question:
{question}
"""

    with st.spinner("Generating answer..."):

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    st.subheader("Answer")
    st.write(response["message"]["content"])

    st.subheader("Sources")

    shown = set()

    for meta in metadatas:
        key = (meta["source"], meta["page"])
        if key not in shown:
            st.write(f"📄 {meta['source']} — Page {meta['page']}")
            shown.add(key)