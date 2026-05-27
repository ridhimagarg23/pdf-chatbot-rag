import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
import google.generativeai as genai

from app import (
    load_pdf,
    create_chunks,
    create_embeddings,
    setup_vector_db,
    answer_question
)

# ----------------------------------------
# Page Configuration
# ----------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ----------------------------------------
# Gemini Setup
# ----------------------------------------

load_dotenv()

api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ----------------------------------------
# PDF Processing Function
# ----------------------------------------

@st.cache_resource
def process_pdf(pdf_path):

    pages = load_pdf(pdf_path)

    chunks, metadatas = create_chunks(
        pages
    )

    embeddings = create_embeddings(
        chunks
    )

    collection = setup_vector_db(
        chunks,
        embeddings,
        metadatas
    )

    return collection, len(chunks)

# ----------------------------------------
# Sidebar
# ----------------------------------------

with st.sidebar:

    st.title("📚 About")

    st.markdown(
        """
        **PDF RAG Chatbot**

        Built using:

        - Streamlit
        - ChromaDB
        - Sentence Transformers
        - Gemini 2.5 Flash
        - RAG Architecture

        Upload a PDF and ask questions
        based on its content.
        """
    )

# ----------------------------------------
# Main UI
# ----------------------------------------

st.title("📄 PDF RAG Chatbot")

st.markdown(
    """
    Upload any PDF and ask questions about
    its content using Retrieval-Augmented Generation (RAG).
    """
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# ----------------------------------------
# PDF Upload Handling
# ----------------------------------------

if uploaded_file:

    st.success("✅ PDF uploaded successfully!")

    st.info(
        f"📂 Uploaded File: {uploaded_file.name}"
    )

    # Save uploaded PDF temporarily

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        pdf_path = temp_file.name

    # Process PDF

    with st.spinner("📖 Processing PDF..."):

        collection, chunk_count = process_pdf(
            pdf_path
        )

    st.success(
        f"✅ PDF processed successfully! ({chunk_count} chunks)"
    )

    st.divider()

    # ----------------------------------------
    # Question Section
    # ----------------------------------------

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("🚀 Get Answer"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = answer_question(
                    question,
                    collection,
                    gemini_model
                )

            st.divider()

            st.subheader("🤖 Answer")

            st.markdown(
                answer
            )