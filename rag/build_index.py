from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_FOLDER = Path(__file__).parent
KNOWLEDGE_BASE_DIR = RAG_FOLDER / "knowledge_base"
PERSIST_DIR = RAG_FOLDER / "chroma_db"


def discover_pdfs(base_dir: Path) -> List[Path]:
    return sorted(p for p in base_dir.glob("**/*.pdf") if p.is_file())


def load_documents(pdf_paths: List[Path]):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


def build_index() -> None:
    load_dotenv()

    pdfs = discover_pdfs(KNOWLEDGE_BASE_DIR)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {KNOWLEDGE_BASE_DIR}. Add PDF files and retry."
        )

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    print(f"Discovered {len(pdfs)} PDF(s). Splitting with size={chunk_size}, overlap={chunk_overlap}.")
    documents = load_documents(pdfs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    splits = splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    embeddings = OpenAIEmbeddings()
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building Chroma index at: {PERSIST_DIR}")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    print("Index built and persisted.")


if __name__ == "__main__":
    build_index()

