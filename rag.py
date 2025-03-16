from typing import List
import faiss
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from utils import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, TOP_K
import os

@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name=LLM_MODEL_NAME, 
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

embedding_model = load_embedding_model()
llm = load_llm()


def retrieve_relevant_chunks(query: str, index: faiss.Index, text_chunks: List[str]) -> List[str]:
    """Retrieves the top K relevant document chunks from FAISS index"""
    if not index or not text_chunks:
        return []

    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        _, indices = index.search(np.array(query_embedding, dtype=np.float32), TOP_K)
        return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        return []

def generate_rag_response(query: str, faiss_index: faiss.Index, text_chunks: List[str]) -> str:
    """Generates response using Retrieval Augmented Generation"""
    if not faiss_index:
        return "No document processed yet"

    try:
        relevant_chunks = retrieve_relevant_chunks(
            query,
            faiss_index,
            text_chunks
        )
        context = "\n\n".join(relevant_chunks)

        prompt_template = PromptTemplate.from_template("""
        Context: {context}
        Question: {query}
        Answer: 
        """)

        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        return llm_chain.run({"context": context, "query": query})
    except Exception as e:
        return f"Error generating response: {str(e)}"