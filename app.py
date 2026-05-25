from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}")

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

model = SentenceTransformer("all-MiniLM-L6-v2") # model loading and we used all-MiniLM-L6-v2 because it is fast, free and good for semantic search

print(f"\nTotal Chunks: {len(chunks)}")

print("\nFIRST CHUNK:\n")
print(chunks[0])

print("\nSECOND CHUNK:\n")
print(chunks[1])

