import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from pathlib import Path
from ingest import ingest_uploaded_pdf

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Data Science PDF Assistant",
    page_icon="📚",
    layout="wide"
)


st.title("📚 Data Science PDF Assistant")

st.write("Chat with the built-in Data Science library or your own uploaded PDFs.")

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# Always get latest collections
# --------------------------------------------------
def get_collections():
    client = chromadb.PersistentClient(path="chroma_db")

    knowledge = client.get_collection("knowledge_collection")
    user = client.get_collection("user_collection")

    return knowledge, user

knowledge_collection, user_collection = get_collections()

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
st.subheader("📤 Upload PDF")

uploaded_pdf = st.file_uploader(
    "Choose a PDF",
    type="pdf"
)

if uploaded_pdf:

    Path("uploads").mkdir(exist_ok=True)

    save_path = Path("uploads") / uploaded_pdf.name

    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    if st.button("Process PDF"):

        with st.spinner("Creating embeddings..."):
            total = ingest_uploaded_pdf(str(save_path))

        st.success(f"Indexed {total} chunks successfully!")
        st.rerun()

st.divider()

# --------------------------------------------------
# Search Mode
# --------------------------------------------------
st.write("### Search In")

search_mode = st.radio(
    "",
    ["📚 Knowledge", "📄 My PDFs", "🌐 Both"],
    horizontal=True
)

st.caption(
    f"Knowledge: {knowledge_collection.count()} chunks | "
    f"My PDFs: {user_collection.count()} chunks"
)

# --------------------------------------------------
# Question
# --------------------------------------------------
question = st.text_input("Ask your question")

# --------------------------------------------------
# Helper Search Function
# --------------------------------------------------
def search_collection(collection, embedding, k=5):

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "document": doc,
            "metadata": meta,
            "distance": dist
        })

    return output


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------
if st.button("Generate Answer") and question:

    # Refresh collections every search
    knowledge_collection, user_collection = get_collections()

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    retrieved = []

    if search_mode == "📚 Knowledge":

        retrieved = search_collection(
            knowledge_collection,
            query_embedding
        )

    elif search_mode == "📄 My PDFs":

        retrieved = search_collection(
            user_collection,
            query_embedding
        )

    else:

        retrieved.extend(
            search_collection(
                knowledge_collection,
                query_embedding
            )
        )

        retrieved.extend(
            search_collection(
                user_collection,
                query_embedding
            )
        )

        retrieved = sorted(
            retrieved,
            key=lambda x: x["distance"]
        )[:5]

    if len(retrieved) == 0:
        st.warning("No documents found.")
        st.stop()

    # Build Context
    context = ""

    for item in retrieved:

        meta = item["metadata"]

        context += f"""
Source: {meta['source']}
Page: {meta['page']}

{item['document']}

------------------------
"""

    # Prompt
    prompt = f"""
You are an expert Data Science tutor.

Answer ONLY from the provided context.

If the answer is unavailable, reply exactly:

I couldn't find it in the provided documents.

Context:
{context}

Question:
{question}
"""

    # LLM
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

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    st.subheader("Answer")

    st.write(response["message"]["content"])

    st.subheader("Sources")

    shown = set()

    with st.expander("📑 View source pages"):

        for item in retrieved:

            meta = item["metadata"]

            key = (meta["source"], meta["page"])

            if key not in shown:

                st.caption(
                    f"📄 {meta['source']} • Page {meta['page']}"
                )

                shown.add(key)