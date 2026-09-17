import os
import docx
import ollama
import chromadb
import pandas as pd
from pypdf import PdfReader
from typing import List, Dict
from bs4 import BeautifulSoup


def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[RAG] File not found: {file_path}")

    ext = file_path.split('.')[-1].lower()
    text = ""

    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        elif ext == 'pdf':
            if PdfReader is None:
                raise ImportError(
                    "[RAG] Please install the 'pypdf' package to read PDF files."
                )

            reader = PdfReader(file_path)
            text = " ".join(
                [
                    page.extract_text()
                    for page in reader.pages
                    if page.extract_text()
                ]
            )

        elif ext == 'docx':
            if docx is None:
                raise ImportError(
                    "[RAG] Please install the 'python-docx' package to read DOCX files."
                )

            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])

        elif ext == 'html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)

        elif ext in ['xlsx', 'csv']:
            if ext == 'csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Convert tabular data into plain text.
            text = df.to_string()

        else:
            raise ValueError(
                f"[RAG] Unsupported file format: '.{ext}'."
            )

    except Exception as e:
        print(f"[RAG] Error reading file {file_path}: {e}")

    return text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        # Move the starting position while keeping an overlap
        # between consecutive chunks to preserve context.
        start = end - overlap

    return chunks


class MemoryRAGSystem:
    def __init__(
        self,
        embed_model: str = 'nomic-embed-text',
        path: str = "./rag_db"
    ):
        self.embed_model = embed_model

        self.chroma_client = chromadb.PersistentClient(path=path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_memory"
        )

        self.chat_history: List[Dict[str, str]] = []

        print(f"[RAG] Embedding model: {self.embed_model}")

    def ingest_document(self, file_path: str):

        print(f"[RAG] Processing file: {file_path}...")
        text = extract_text_from_file(file_path)

        if not text.strip():
            print("[RAG] File is empty or no text could be extracted.")
            return

        chunks = chunk_text(text)
        print(
            f"[RAG] Split document into {len(chunks)} chunks. "
            "Generating embeddings..."
        )

        for i, chunk in enumerate(chunks):

            # Generate an embedding vector for each chunk using Ollama.
            response = ollama.embeddings(
                model=self.embed_model,
                prompt=chunk
            )
            embedding = response['embedding']

            # Generate a unique ID for each document chunk.
            doc_id = f"{os.path.basename(file_path)}_chunk_{i}"

            # Store the chunk and its embedding in ChromaDB.
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": file_path}]
            )

        print("[RAG] Document successfully added to memory!\n")

    def search(self, question: str) -> str:

        # Generate an embedding for the user's query.
        # GPU usage is disabled here to reduce VRAM consumption.
        embed_response = ollama.embeddings(
            model=self.embed_model,
            prompt=question,
            options={"num_gpu": 0}
        )
        query_embedding = embed_response['embedding']

        # Search ChromaDB for the three most relevant document chunks.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        if results['documents'] and len(results['documents'][0]) > 0:
            answers = "\n\n".join(results['documents'][0])
            return answers

        return ""

