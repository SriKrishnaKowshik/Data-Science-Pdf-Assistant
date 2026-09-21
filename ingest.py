from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# Paths
# -----------------------------
DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"
COLLECTION_NAME = "pdf_collection"

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Create Chroma database
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_FOLDER)

# Delete old collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# -----------------------------
# Read PDF
# -----------------------------
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append((page_number, text))

    return pages

# -----------------------------
# Chunk text
# -----------------------------
def chunk_text(text, chunk_size=250, overlap=40):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -----------------------------
# Process all PDFs
# -----------------------------
pdf_files = list(Path(DATA_FOLDER).glob("*.pdf"))

if not pdf_files:
    print("No PDF found inside data folder.")
    exit()

doc_id = 0

for pdf in pdf_files:
    print(f"Processing: {pdf.name}")

    pages = read_pdf(pdf)

    for page_number, page_text in pages:

        chunks = chunk_text(page_text)

        for chunk in chunks:

            embedding = model.encode(
                chunk,
                normalize_embeddings=True
          ).tolist()

            collection.add(
                ids=[str(doc_id)],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "page": page_number,
                    "source": pdf.name
                }]
            )

            doc_id += 1

print("Finished! Data stored in ChromaDB.")