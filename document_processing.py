import fitz  # PyMuPDF
from io import BytesIO
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from utils import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

embedding_model = load_embedding_model()

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """Extracts text from PDF documents using PyMuPDF with error handling"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return ""

def chunk_text(text: str) -> List[str]:
    """Splits text into meaningful chunks with overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    return splitter.split_text(text)

def create_faiss_index(text_chunks: List[str]) -> faiss.Index:
    """Creates and returns FAISS index with embeddings"""
    if not text_chunks:
        return None

    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        return index
    except Exception as e:
        st.error(f"Index creation failed: {str(e)}")
        return None