from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# --------------------------------------------------
# Folders
# --------------------------------------------------
KNOWLEDGE_FOLDER = "knowledge"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# PDF Reader
# --------------------------------------------------
def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append((page_num, text))

    return pages

# --------------------------------------------------
# Chunking
# --------------------------------------------------
def chunk_text(text, chunk_size=250, overlap=40):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunks.append(" ".join(words[start:end]))

        start += chunk_size - overlap

    return chunks

# --------------------------------------------------
# Create Collections
# --------------------------------------------------
def create_collections():

    client = chromadb.PersistentClient(path="chroma_db")

    # Delete old collections
    for name in ["knowledge_collection", "user_collection"]:
        try:
            client.delete_collection(name)
        except:
            pass

    knowledge = client.create_collection(
        name="knowledge_collection",
        metadata={"hnsw:space": "cosine"}
    )

    user = client.create_collection(
        name="user_collection",
        metadata={"hnsw:space": "cosine"}
    )

    return knowledge, user

# --------------------------------------------------
# Index Knowledge PDFs
# --------------------------------------------------
def index_folder(folder, collection):

    doc_id = 0

    pdfs = list(Path(folder).glob("*.pdf"))

    for pdf in pdfs:

        print(f"Indexing {pdf.name}")

        pages = read_pdf(pdf)

        for page_num, page_text in pages:

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
                        "page": page_num
                    }]
                )

                doc_id += 1

# --------------------------------------------------
# Index Uploaded PDF (ONLY ONE PDF)
# --------------------------------------------------
def ingest_uploaded_pdf(pdf_path):

    client = chromadb.PersistentClient(path="chroma_db")

    # Replace previous uploaded PDF
    try:
        client.delete_collection("user_collection")
    except:
        pass

    collection = client.create_collection(
        name="user_collection",
        metadata={"hnsw:space": "cosine"}
    )

    pages = read_pdf(pdf_path)

    doc_id = 0

    for page_num, page_text in pages:

        chunks = chunk_text(page_text)

        for chunk in chunks:

            embedding = model.encode(
                chunk,
                normalize_embeddings=True
            ).tolist()

            collection.add(
                ids=[f"user_{doc_id}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": Path(pdf_path).name,
                    "page": page_num
                }]
            )

            doc_id += 1

    return collection.count()

# --------------------------------------------------
# Build Knowledge Database
# --------------------------------------------------
if __name__ == "__main__":

    knowledge_collection, _ = create_collections()

    index_folder(KNOWLEDGE_FOLDER, knowledge_collection)

    print("\nKnowledge Database Ready!")
    print(f"Knowledge chunks: {knowledge_collection.count()}")
    print("User PDFs are indexed only from the Streamlit UI.")