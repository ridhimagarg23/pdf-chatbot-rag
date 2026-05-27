# download necessary libraries and modules 

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import os
from dotenv import load_dotenv
import google.generativeai as genai

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# functions 

def load_pdf(pdf_path):  # function to load the pdf and extract text from it (page wise)

    reader = PdfReader(pdf_path)

    pages = []

    for page_num, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append(
                {
                    "page": page_num,
                    "text": text
                }
            )

    return pages


def create_chunks(text):  # function to create chunks of the extracted text from the pdf
    
    ''' 
    using RecursiveCharacterTextSplitter from langchain_text_splitters that allows us to 
    split the text into smaller chunks while maintaining the context and meaning of the text. 
    we set the chunk size to 500 characters and an overlap of 100 characters to ensure that 
    we don't lose important information between chunks.

    '''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    metadatas = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            metadatas.append(
                {
                    "page": page["page"]
                }
            )

    return chunks, metadatas

def create_embeddings(chunks):  # function to create embeddings by model loading and we used all-MiniLM-L6-v2 because it is fast, free and good for semantic search
    
    embeddings = embedding_model.encode(chunks)
    
    return embeddings

def setup_vector_db(chunks, embeddings, metadatas):  # function to setup the vector database using ChromaDB, we create a collection named "pdf_chunks" and add the chunks along with their corresponding embeddings to the collection. Each chunk is assigned a unique ID for easy retrieval during querying.

    client = chromadb.Client()

    try:
        client.delete_collection("pdf_chunks")
    except:
        pass

    collection = client.create_collection(
        name="pdf_chunks"
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
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

    # Create embedding for user's question
    question_embedding = embedding_model.encode(
        [question]
    )

    # Retrieve most relevant chunks
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=20
    )

    print("\n===== RETRIEVED CHUNKS =====\n")

    for i, (chunk, distance) in enumerate(
        zip(
            results["documents"][0],
            results["distances"][0]
        ),
        start=1
    ):

        print(
            f"\n------ CHUNK {i} | Distance: {distance:.4f} ------\n"
        )

        print(chunk[:500])

    # Combine retrieved chunks into one context
    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(
        retrieved_chunks
    )

    # Prompt for Gemini
    prompt = f"""
    You are a helpful assistant.
    Answer the question ONLY using the provided context.
    
    If the answer exists in the context,
    provide the exact answer.
    
    If the answer cannot be found in the context,
    reply with:
    
    "I could not find this information in the document."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    response = gemini_model.generate_content(
        prompt
    )

    return response.text

# main flow of the code

if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    # Load PDF page by page
    pages = load_pdf(
        pdf_path
    )

    # Create chunks and page metadata
    chunks, metadatas = create_chunks(
        pages
    )

    # Debug check
    for i, chunk in enumerate(chunks):

        if "ridhima" in chunk.lower():

            print("\n===== FOUND CHUNK =====\n")

            print(chunk)

    # Create embeddings
    embeddings = create_embeddings(
        chunks
    )

    # Store in ChromaDB
    collection = setup_vector_db(
        chunks,
        embeddings,
        metadatas
    )

    # Gemini model
    gemini_model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    # Ask question
    question = input(
        "\n\nAsk a question:\n"
    )

    answer = answer_question(
        question,
        collection,
        gemini_model
    )

    print("\n===== ANSWER =====\n")

    print(answer)