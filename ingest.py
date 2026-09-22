from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# Folders
# -----------------------------
KNOWLEDGE_FOLDER = "knowledge"
UPLOAD_FOLDER = "uploads"

# -----------------------------
# Embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Chroma
# -----------------------------
client = chromadb.PersistentClient(path="chroma_db")

# Remove old collections
for name in ["knowledge_collection", "user_collection"]:
    try:
        client.delete_collection(name)
    except:
        pass

knowledge_collection = client.create_collection(
    name="knowledge_collection",
    metadata={"hnsw:space": "cosine"}
)

user_collection = client.create_collection(
    name="user_collection",
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
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=250, overlap=40):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(" ".join(words[start:end]))

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Index one folder
# -----------------------------
def index_folder(folder, collection):

    pdfs = list(Path(folder).glob("*.pdf"))

    doc_id = 0

    for pdf in pdfs:

        print(f"Indexing {pdf.name}")

        pages = read_pdf(pdf)

        for page_number, page_text in pages:

            chunks = chunk_text(page_text)

            for chunk in chunks:

                embedding = model.encode(
                    chunk,
                    normalize_embeddings=True
                ).tolist()

                collection.add(
                    ids=[f"{pdf.stem}_{doc_id}"],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": pdf.name,
                        "page": page_number
                    }]
                )

                doc_id += 1


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    index_folder(KNOWLEDGE_FOLDER, knowledge_collection)
    index_folder(UPLOAD_FOLDER, user_collection)

    print("\nKnowledge chunks :", knowledge_collection.count())
    print("User chunks      :", user_collection.count())