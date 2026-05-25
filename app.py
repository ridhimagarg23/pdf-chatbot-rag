import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

# print(f"Total Pages: {len(reader.pages)}")  used to print the total number of pages in the pdf

full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(full_text)
ids = [f"chunk_{i}" for i in range(len(chunks))]

model = SentenceTransformer("all-MiniLM-L6-v2") # model loading and we used all-MiniLM-L6-v2 because it is fast, free and good for semantic search
client = chromadb.Client()

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

embedding = model.encode(chunks) 

collection = client.create_collection(
    name="pdf_chunks"
)

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embedding.tolist()
)

# print(collection.count())  prints total number of chunks in the collection

question = input("\n\n Ask a question: \n")

results = collection.query(
    query_texts=[question],
    n_results=4
)

retrieved_chunks = results["documents"][0]
context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

response = gemini_model.generate_content(prompt)
print(response.text)