from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
import os

@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name="llama3-70b-8192", 
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

llm = load_llm()

def generate_summary(text: str) -> str:
    """Robust summary generation with error handling"""
    if not text:
        return "No content to summarize"

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        docs = text_splitter.create_documents([text])

        map_template = """Summarize this legal document chunk:
        {docs}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate.from_template(map_template)

        reduce_template = """Combine these summaries:
        {doc_summaries}
        FINAL SUMMARY:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )

        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs"
        ).run(docs)
    except Exception as e:
        return f"Summary generation failed: {str(e)}"