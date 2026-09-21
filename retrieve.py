import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Connect to ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_collection")

# -----------------------------
# User question
# -----------------------------
question = input("Ask a question: ")

# Convert question to embedding
query_embedding = model.encode(question).tolist()

# Search similar chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# -----------------------------
# Display results
# -----------------------------
print("\nTop 3 Relevant Chunks\n")

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i in range(len(documents)):
    print("=" * 60)
    print(f"Result {i+1}")
    print(f"Source : {metadatas[i]['source']}")
    print(f"Page   : {metadatas[i]['page']}")
    print(f"Distance: {distances[i]:.4f}")
    print("-" * 60)
    print(documents[i][:500])  # first 500 characters
    print()