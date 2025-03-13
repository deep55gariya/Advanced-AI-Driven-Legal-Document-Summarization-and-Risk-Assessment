import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import base64
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import difflib
import requests
from bs4 import BeautifulSoup
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

st.set_page_config(
    page_title="LegalDoc Analyst",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize NLP resources
import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data")) # Or your preferred path

try:
    nltk.download('vader_lexicon', download_dir=os.path.join(os.getcwd(), "nltk_data")) #optional
except LookupError as e:
    st.error(f"Error downloading NLTK resources: {e}.  Please check your internet connection and try again.")
    st.stop()

from nltk.sentiment import SentimentIntensityAnalyzer


from document_processing import extract_text_from_pdf
from document_processing import chunk_text, create_faiss_index
from rag import generate_rag_response
from risk_analysis import advanced_risk_assessment, visualize_risks
from summarization import generate_summary
from comparison import compare_documents
from compliance import fetch_compliance_guidelines
from report_generation import generate_pdf, send_email
from utils import initialize_session_state

# Initialize session state
initialize_session_state()


def main():
    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f5f7fb;}
        .stButton>button {border-radius: 8px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {width: 100%;}
        .stExpander .st-emotion-cache-1hynsf2 {border-radius: 10px;}
        .metric-box {padding: 20px; border-radius: 10px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .risk-critical {color: #dc3545!important;}
        .risk-high {color: #ff6b6b!important;}
        .risk-medium {color: #ffd93d!important;}
        .risk-low {color: #6c757d!important;}

    </style>
    """, unsafe_allow_html=True)

    # App Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    with col2:
        st.title("Legal Document Summarizer and Analyzer")
        st.markdown("**AI-powered Contract Analysis & Risk Assessment**")

    # Main Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Analysis",
        "📊 Risk Dashboard",
        "🔀 Comparison",
        "📜 Compliance",
        "📧 Report"
    ])

    # Document Processing Section
    with tab1:
        st.header("Document Processing")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])
            if uploaded_file and not st.session_state.document_processed:
                if st.button("Analyze Document", type="primary"):
                    with st.status("Processing document...", expanded=True) as status:
                        try:
                            st.write("Extracting text...")
                            st.session_state.full_text = extract_text_from_pdf(uploaded_file)

                            st.write("Chunking text...")
                            st.session_state.text_chunks = chunk_text(st.session_state.full_text)

                            st.write("Creating search index...")
                            st.session_state.faiss_index = create_faiss_index(st.session_state.text_chunks)

                            st.write("Generating summary...")
                            st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)

                            st.write("Assessing risks...")
                            st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)

                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            st.session_state.document_processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.session_state.document_processed = False

        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Document Summary")
                st.write(st.session_state.summaries.get('document', "No summary available"))

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "Download Text Summary",
                        data=st.session_state.summaries.get('document', ""),
                        file_name="document_summary.txt",
                        use_container_width=True
                    )
                with col_d2:
                    if st.button("Generate Full Report PDF", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            pdf_buffer = generate_pdf(
                                st.session_state.summaries['document'],
                                st.session_state.risk_data
                            )
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("PDF ready for download!")

    # Risk Dashboard
    if st.session_state.document_processed:
        with tab2:
            st.header("Risk Analysis Dashboard")
            risk_data = st.session_state.risk_data

            # Risk Metrics
            with st.container(border=True):
                cols = st.columns(4)
                metric_style = """
                    <style>
                        .metric-box {
                            padding: 20px;
                            border-radius: 10px;
                            background: white;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            height: 150px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                        }
                        .metric-title {
                            font-size: 1.1rem;
                            margin-bottom: 8px;
                            font-weight: 600;
                            color: #666;
                        }
                        .metric-value {
                            font-size: 2.5rem;
                            font-weight: 700;
                            line-height: 1.2;
                            color: #dc3545 !important;
                        }
                        .metric-subtext {
                            font-size: 1rem;
                            color: #666;
                        }
                        .risk-critical { color: #dc3545; }
                        .risk-high { color: #ff6b6b; }
                    </style>
                """
                st.markdown(metric_style, unsafe_allow_html=True)

                # Overall Risk Score
                with cols[0]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-critical">Overall Risk Score</div>
                            <div class="metric-value risk-critical">
                                {risk_data.get("total_score", 0)}
                                <span class="metric-subtext">/100</span>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # Total Risks
                with cols[1]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title">Total Risks</div>
                            <div class="metric-value">
                                {risk_data.get("total_risks", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # Critical Risks
                with cols[2]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-critical">Critical Risks</div>
                            <div class="metric-value risk-critical">
                                {risk_data["severity_counts"].get("Critical", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                # High Risks
                with cols[3]:
                    st.markdown(f'''
                        <div class="metric-box">
                            <div class="metric-title risk-high">High Risks</div>
                            <div class="metric-value risk-high">
                                {risk_data["severity_counts"].get("High", 0)}
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

            # Visualizations
            with st.container(border=True):
                fig1, fig2 = visualize_risks(risk_data)
                if fig1 and fig2:
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col_v2:
                        st.plotly_chart(fig2, use_container_width=True)

            # Detailed Risk Breakdown
            with st.container(border=True):
                st.subheader("Risk Category Breakdown")
                if risk_data.get('categories'):
                    df = pd.DataFrame.from_dict(risk_data['categories'], orient='index')
                    st.dataframe(
                        df,
                        column_config={
                            "score": st.column_config.ProgressColumn(
                                "Score",
                                help="Risk score (0-40)",
                                format="%f",
                                min_value=0,
                                max_value=40,
                            )
                        },
                        use_container_width=True
                    )

    # Document Comparison
    with tab3:
        st.header("Document Comparison")
        if st.session_state.document_processed:
            with st.container(border=True):
                compare_file = st.file_uploader("Upload Comparison Document", type=["pdf"])
                if compare_file:
                    try:
                        compare_text = extract_text_from_pdf(compare_file)
                        comparison = compare_documents(st.session_state.full_text, compare_text)
                        st.markdown(
                            f'<div style="border:1px solid #eee; padding:20px; border-radius:8px">'
                            f'{comparison}</div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")

    # Compliance Section
    with tab4:
        st.header("Compliance Checklists")
        guidelines = fetch_compliance_guidelines()

        if isinstance(guidelines, dict) and "error" not in guidelines:
            for regulation, data in guidelines.items():
                with st.expander(f"🔍 {regulation} Compliance Checklist"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"{regulation} Requirements")
                        for item in data.get("checklist", []):
                            st.markdown(f"- {item}")

                    with col2:
                        st.download_button(
                            label=f"Download {regulation} Checklist",
                            data="\n".join(data.get("checklist", [])),
                            file_name=f"{regulation}_checklist.txt",
                            use_container_width=True
                        )

                    if data.get("latest_updates"):
                        st.markdown("---")
                        st.caption(f"**Latest {regulation} Updates**")
                        for update in data["latest_updates"][:3]:
                            st.markdown(f"📢 {update}")
        else:
            st.error("Failed to load compliance guidelines")

    # Report Section
    with tab5:
        st.header("Report Generation")
        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Email Report")
                email = st.text_input("Recipient Email Address", placeholder="legal@company.com")

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    if st.button("📧 Send Email Report", use_container_width=True):
                        if email and st.session_state.get('pdf_buffer'):  # Check if pdf_buffer exists
                            if send_email(email, st.session_state.pdf_buffer):
                                st.success("Report sent successfully!")
                            else:
                                st.error("Failed to send email")
                        else:
                            st.warning("Please generate PDF first and enter valid email")

                with col_e2:
                    if st.session_state.get('pdf_buffer'):  # Check if pdf_buffer exists
                        st.download_button(
                            label="⬇️ Download Full Report",
                            data=st.session_state.pdf_buffer,
                            file_name="Legal_Analysis_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

    # Chat Interface (Floating in Sidebar)
    if st.session_state.document_processed:
        with st.sidebar:
            st.header("Document Q&A")
            for role, msg in st.session_state.chat_history[-3:]:
                with st.chat_message(role):
                    st.write(msg)

            if query := st.chat_input("Ask about the document..."):
                with st.spinner("Analyzing..."):
                    response = generate_rag_response(query, st.session_state.faiss_index, st.session_state.text_chunks)
                    st.session_state.chat_history.extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()
            # License Footer
            st.markdown("---")
            st.caption("""
                © 2025 VidzAI - All Rights Reserved
                This software is proprietary and confidential
                Unauthorized use or distribution prohibited
            """)

if __name__ == "__main__":
    main()
