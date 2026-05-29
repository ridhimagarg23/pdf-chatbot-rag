# 📄 PDF RAG Chatbot

An intelligent AI-powered PDF Question Answering system built using Retrieval-Augmented Generation (RAG).

This project allows users to upload PDF documents and ask questions directly from the document content using semantic search and local Large Language Models (LLMs).

---

# 🚀 Features

✅ Upload any PDF document
✅ Extract text page-by-page
✅ Smart semantic chunking
✅ Vector embeddings using Sentence Transformers
✅ ChromaDB vector storage
✅ Semantic similarity retrieval
✅ Local AI-powered answer generation
✅ Fully local setup (No paid APIs required)
✅ Streamlit interactive UI screen

---

# 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) combines:

* **Information Retrieval**
* **Vector Search**
* **Large Language Models**

to generate accurate answers grounded in external documents.

Instead of relying only on model memory, the chatbot retrieves relevant chunks from the uploaded PDF before generating an answer.

---

# ⚡ Project Workflow

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunk Creation
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
Semantic Retrieval
    ↓
Local LLM Generation
    ↓
Final Answer
```

---

# 🛠️ Tech Stack

| Technology               | Purpose              |
| ------------------------ | -------------------- |
| Streamlit                | Frontend UI          |
| ChromaDB                 | Vector Database      |
| Sentence Transformers    | Embedding Generation |
| LangChain                | Text Chunking        |
| HuggingFace Transformers | Local LLM            |
| TinyLlama / FLAN-T5      | Text Generation      |
| PyPDF                    | PDF Text Extraction  |

---

# 📂 Project Structure

```text
pdf-chatbot-rag/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample.pdf
│
└── chroma_db/
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ridhimagarg23/pdf-chatbot-rag.git
cd pdf-chatbot-rag
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

---

# 💬 How to Use

1. Launch the Streamlit app
2. Upload a PDF document
3. Wait for processing
4. Ask questions related to the PDF
5. Get AI-generated answers from the document context

---

# 🔍 Core Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Embedding Models
* Large Language Models (LLMs)
* Prompt Engineering

---

# 🔥 Future Improvements

* Multi-PDF Support
* Chat Memory
* Conversational RAG
* OCR Support for Scanned PDFs
* Source Citations
* Hybrid Search (Keyword + Semantic)
* Better Embedding Models
* Reranking Pipeline

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository and submit pull requests.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

### Ridhima Garg

AI/ML Enthusiast • Generative AI • RAG Systems • LLM Applications
