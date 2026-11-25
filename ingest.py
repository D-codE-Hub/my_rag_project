import os
import hashlib
import fitz
import chromadb

from sentence_transformers import SentenceTransformer

# 1. Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to persistent ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="company_documents"
)


# 3. Read TXT and PDF files
def read_document(file_path):

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    elif file_path.endswith(".pdf"):
        pdf = fitz.open(file_path)

        return "\n".join(
            page.get_text()
            for page in pdf
        )

    return ""


# 4. Split text into chunks
def split_text(text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 5. Process all documents
def ingest_documents():

    folder = "./documents"

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        if not filename.endswith((".txt", ".pdf")):
            continue

        print("Processing:", filename)

        text = read_document(file_path)

        chunks = split_text(text)

        if not chunks:
            continue

        # Generate embeddings
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        ids = [
            hashlib.sha256(
                f"{filename}:{i}".encode()
            ).hexdigest()
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": filename,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        # Save into ChromaDB
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print("Stored chunks:", len(chunks))

    print("\nDocument ingestion completed!")


if __name__ == "__main__":
    ingest_documents()