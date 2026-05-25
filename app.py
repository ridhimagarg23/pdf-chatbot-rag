# download necessary libraries and modules 

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

# functions 

def load_pdf(pdf_path):  # function to load the pdf and extract text from it

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            full_text += text + "\n"

    return full_text


def create_chunks(text):  # function to create chunks of the extracted text from the pdf
    
    ''' 
    using RecursiveCharacterTextSplitter from langchain_text_splitters that allows us to 
    split the text into smaller chunks while maintaining the context and meaning of the text. 
    we set the chunk size to 500 characters and an overlap of 100 characters to ensure that 
    we don't lose important information between chunks.

    '''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks

def create_embeddings(chunks):  # function to create embeddings by model loading and we used all-MiniLM-L6-v2 because it is fast, free and good for semantic search

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    return embeddings 

def setup_vector_db(chunks, embeddings):  # function to setup the vector database using ChromaDB, we create a collection named "pdf_chunks" and add the chunks along with their corresponding embeddings to the collection. Each chunk is assigned a unique ID for easy retrieval during querying.
  
    client = chromadb.Client()

    collection = client.create_collection(
        name="pdf_chunks"
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return collection

def answer_question(question, collection, gemini_model):
    """
    Retrieves relevant chunks and generates an answer using Gemini.

    Args:
        question (str): User question.
        collection: Chroma collection.
        gemini_model: Gemini model instance.

    Returns:
        str: Generated answer.
    """

    results = collection.query(
        query_texts=[question],
        n_results=8
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

    return response.text

# main flow of the code

pdf_path = "data/sample.pdf"
full_text = load_pdf(pdf_path)

# print(f"Total Pages: {len(reader.pages)}")  used to print the total number of pages in the pdf

chunks = create_chunks(full_text)

embeddings = create_embeddings(chunks)

collection = setup_vector_db(
    chunks,
    embeddings
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

question = input("\n\n Ask a question: \n")

answer = answer_question(
    question,
    collection,
    gemini_model
)

print(answer)