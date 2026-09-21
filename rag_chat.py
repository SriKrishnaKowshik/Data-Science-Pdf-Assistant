import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# -------------------------
# Load embedding model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Connect ChromaDB
# -------------------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_collection")

print("=" * 50)
print("📚 Data Science PDF Assistant")
print("=" * 50)

# -------------------------
# Ask user
# -------------------------
question = input("\nAsk your question: ")

# Convert question to vector
query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# Retrieve top 5 chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]


# ----- This helps you debug whether ChromaDB retrieved the correct pages.------
#print("\nRetrieved Context:\n")

#for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
 #   print(doc[:350])      # First 350 characters
 #   print(f"Result {i} | {meta['source']} | Page {meta['page']}")
 #  print()
 #   print("=" * 50)


# -------------------------
# Build context
# -------------------------
context = ""

for doc, meta in zip(documents, metadatas):
    context += f"""
Source: {meta['source']}
Page: {meta['page']}

{doc}

------------------------
"""

# -------------------------
# Prompt for Gemma
# -------------------------
prompt = f"""
You are an expert Machine Learning tutor.

Use ONLY the retrieved context to answer.

Instructions:
- Give a clear definition first.
- Explain in 2–4 short paragraphs.
- Include an intuition or example if present in the context.
- Do not mention searching PDFs.
- If the answer is missing, reply exactly:
  I couldn't find it in the uploaded PDFs.

Context:
{context}

Question:
{question}
"""

# -------------------------
# Generate answer
# -------------------------
# Generate answer
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# -------------------------
# Display answer
# -------------------------
print("\n" + "=" * 50)
print("ANSWER")
print("=" * 50)

print(response["message"]["content"])

print("\nSources Used:")
for meta in metadatas:
    print(f"• {meta['source']} (Page {meta['page']})")