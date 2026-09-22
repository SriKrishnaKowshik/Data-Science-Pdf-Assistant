# 📚 Data Science PDF Assistant

A **local Retrieval-Augmented Generation (RAG)** application that allows users to chat with a built-in **Data Science knowledge base** or their own uploaded PDFs using **ChromaDB, Sentence Transformers, Ollama, and Streamlit**.

> No OpenAI API key required — everything runs locally.

---

## 🚀 Features

- 📖 Built-in Data Science knowledge base (ML books, lecture notes)
- 📄 Upload your own PDF and chat with it
- 🌐 Hybrid search (Knowledge / My PDFs / Both)
- 🧠 Semantic search using ChromaDB + Sentence Transformers
- 🤖 Local LLM with Ollama (Llama 3.2 / Gemma)
- 📑 Page-level source citations
- 💻 Streamlit web interface
- 🔒 Fully offline & privacy-friendly

---

## 🖼️ Demo

> Add screenshots of your Streamlit UI here after uploading them to GitHub.

| Home | Answer |
|------|------|
| `assets/home.png` | `assets/answer.png` |

---

## 🏗️ Architecture

```text
                    User Question
                          │
                          ▼
                Sentence Transformer
                  (Embedding Model)
                          │
                          ▼
                 ChromaDB Vector Search
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
  Knowledge Collection            User Collection
 (Built-in DS PDFs)             (Uploaded PDFs)
          │                               │
          └───────────────┬───────────────┘
                          ▼
                  Retrieved Context
                          │
                          ▼
                 Ollama Local LLM
                    (Llama 3.2)
                          │
                          ▼
                 Answer + Source Pages
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core application |
| Streamlit | Web UI |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| Ollama | Local LLM inference |
| PyPDF | PDF text extraction |

---

## 📂 Project Structure

```text
Data_Science_PDF_Assistant/
│
├── app.py                 # Streamlit UI
├── ingest.py              # Knowledge & upload indexing
├── rag_chat.py            # CLI RAG chat
├── retrieve.py            # Retrieval testing
├── check_db.py            # Verify database
├── requirements.txt
│
├── knowledge/             # Built-in Data Science PDFs
│   ├── ML_Book.pdf
│   └── ISLP_Book.pdf
│
├── uploads/               # User uploaded PDFs
│
├── chroma_db/             # Knowledge vector database
└── user_db/               # User vector database
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Data-Science-PDF-Assistant.git
cd Data-Science-PDF-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com

Pull the model:

```bash
ollama pull llama3.2:3b
```

---

## 📚 Build the Knowledge Base

Place your Data Science PDFs inside the `knowledge/` folder.

Run:

```bash
python ingest.py
```

Example output:

```text
Knowledge Database Ready!
Knowledge chunks: 2060
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

## 💡 How to Use

### Built-in Knowledge

1. Run `ingest.py`
2. Select **Knowledge**
3. Ask questions about Machine Learning, Statistics, etc.

### Upload Your Own PDF

1. Open the Streamlit app
2. Upload a PDF
3. Click **Process PDF**
4. Select **My PDFs**
5. Start chatting with the document

### Hybrid Search

Choose **Both** to retrieve information from:

- Built-in Data Science books
- Your uploaded PDF

---

## 📈 Current Capabilities

| Feature | Status |
|----------|--------|
| Local RAG | ✅ |
| Semantic Search | ✅ |
| ChromaDB | ✅ |
| PDF Upload | ✅ |
| Hybrid Retrieval | ✅ |
| Source Citations | ✅ |
| Offline LLM | ✅ |

---

## 🎯 Future Improvements

- [ ] ChatGPT-style conversation interface
- [ ] Streaming responses
- [ ] Clickable source preview
- [ ] BM25 + Semantic hybrid retrieval
- [ ] Multiple uploaded PDFs
- [ ] Conversation memory

---

## 👨‍💻 Author

**Sri Krishna Kowshik**

M.Sc. Computer Science — Technische Universität Dresden

- Interests: Data Science, Machine Learning, Generative AI

---

## ⭐ If you like this project

Consider giving the repository a **Star** ⭐
