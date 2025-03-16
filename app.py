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
from comparison import compare_documents, export_comparison_report, compare_documents_tabular
from compliance import fetch_updates_for_document, classify_document_type, fetch_document_compliance
from report_generation import generate_pdf, send_email, create_email_text
from utils import initialize_session_state

# Initialize session state
initialize_session_state()


# ------------------------------ UTILITY FUNCTIONS ------------------------------

def validate_email(email):
    """Basic email validation."""
    import re
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email)

# ------------------------------ CUSTOM CSS ------------------------------

CUSTOM_CSS = """
<style>
    /*  General Theme */
    :root {
        --primary-color: #2962FF;  /* A professional blue */
        --secondary-color: #64B5F6; /* A lighter blue */
        --text-color: #333333;
        --background-color: #F5F7FA;
        --card-background: #FFFFFF;
        --border-color: #E0E0E0;
    }

    body {
        font-family: 'Roboto', sans-serif; /* Example: Use a professional font */
        color: var(--text-color);
        background-color: var(--background-color);
    }

    .main {
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 10px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color);
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 25px; /* Rounded buttons */
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 500;
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }

    .stButton>button:hover {
        background-color: var(--secondary-color);
    }

    /* File Uploader */
    .stFileUploader label {
        background-color: var(--card-background);
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
    }

    /* Metric Boxes */
    .metric-box {
        background-color: var(--card-background);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    .metric-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #555;
    }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    /* Expander */
    .stExpander {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .stExpander>label {
        font-weight: 500;
    }

    /* Download Button */
    .stDownloadButton>button {
        background-color: #4CAF50;  /* A professional green */
        color: white;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 500;
        transition: background-color 0.3s ease;
        width: 100%;
    }

    .stDownloadButton>button:hover {
        background-color: #388E3C;
    }

    /* Dataframe */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 10px;
        background-color: var(--card-background);
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] > div {
        gap: 1rem;
    }

    div[data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    div[data-baseweb="tab"]:hover {
        background-color: var(--secondary-color);
        color: white;
    }

    div[aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 0.75rem 1rem;
        margin-bottom: 10px;
    }

    .stChatMessage[data-testid="stChatMessageContent"]:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        border-width: 10px;
        border-style: solid;
    }

    .stChatMessage[data-testid="stChatMessageContent"][data-streamlit-theme="dark"]:before {
        border-color: transparent transparent var(--background-color) var(--background-color);
        left: auto;
        right: 0;
    }

    .stChatMessage[data-testid="stChatMessageContent"][data-streamlit-theme="light"]:before {
        border-color: transparent transparent var(--background-color) var(--background-color);
    }

    /* Add more CSS as needed */

</style>
"""

# ------------------------------ MAIN APP ------------------------------

def main():
    # Apply Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # App Header with Logo and Title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("A_professional_logo_for_'Legal_Document_Analysis_a.png", width=120) # Larger logo
    with col2:
        st.title("AI-Powered Legal Document Analysis")
        st.markdown("© 2025 VidzAI - All Rights Reserved. Confidential.")

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📄 Document Analysis",
        "📊 Risk Dashboard",
        "❓ Q&A Chat",
        "🔀 Comparison",
        "📜 Compliance",
        "🔍 Legal Updates",
        "📧 Email Report"
    ])

    # ------------------------------ TAB 1: Document Analysis ------------------------------

    with tab1:
        st.header("Document Analysis")
        st.markdown("Upload a legal document (PDF) to begin the analysis.")

        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])

            if uploaded_file and not st.session_state.document_processed:
                if st.button("Analyze Document", type="primary"):
                    with st.status("Analyzing Document...", expanded=True) as status:
                        try:
                            # Document Processing Steps
                            st.write("1. Extracting text from PDF...")
                            st.session_state.full_text = extract_text_from_pdf(uploaded_file)

                            st.write("2. Chunking text into smaller segments...")
                            st.session_state.text_chunks = chunk_text(st.session_state.full_text)

                            st.write("3. Creating a search index for efficient querying...")
                            st.session_state.faiss_index = create_faiss_index(st.session_state.text_chunks)

                            st.write("4. Generating a concise summary of the document...")
                            st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)

                            st.write("5. Performing advanced risk assessment...")
                            st.session_state.risk_data = advanced_risk_assessment(st.session_state.full_text)

                            st.write("6. Classifying the document type...")
                            st.session_state.document_categories = classify_document_type(st.session_state.full_text)

                            st.write("7. Fetching relevant legal updates...")
                            st.session_state.legal_updates = fetch_updates_for_document(st.session_state.full_text)

                            st.write("8. Analyzing compliance requirements...")
                            st.session_state.document_compliance = fetch_document_compliance(st.session_state.full_text)

                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            st.success("Document analysis complete!  See results in other tabs.")
                            st.session_state.document_processed = True
                            st.rerun()  # Refresh the app to show results

                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.session_state.document_processed = False

        # Display Summary and Options if Document is Processed
        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Document Summary")
                st.write(st.session_state.summaries.get('document', "No summary available"))

                if st.session_state.document_categories:
                    st.subheader("Document Classification")
                    for category, confidence in st.session_state.document_categories[:3]:
                        st.write(f"- {category}: {confidence:.2f} confidence")

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
                                st.session_state.risk_data,
                                st.session_state.legal_updates if hasattr(st.session_state, 'legal_updates') else None,
                                st.session_state.document_compliance if hasattr(st.session_state, 'document_compliance') else None
                            )
                            st.session_state.pdf_buffer = pdf_buffer
                            st.success("PDF ready for download!")
        else:
            st.info("Upload a document to begin the analysis.")
    # ------------------------------ TAB 2: Risk Dashboard ------------------------------

    with tab2:
        st.header("Risk Analysis Dashboard")

        if st.session_state.document_processed:
            risk_data = st.session_state.risk_data

            # Risk Metrics
            with st.container(border=True):
                st.subheader("Key Risk Metrics")
                cols = st.columns(4)

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
                st.subheader("Risk Visualizations")
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
        else:
            st.info("Please upload and analyze a document to see risk analysis.")

    # ------------------------------ TAB 3: Q&A Chat ------------------------------

    with tab3:
        st.header("Document Q&A")

        if st.session_state.document_processed:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for role, msg in st.session_state.chat_history:
                    with st.chat_message(role):
                        st.write(msg)

            # Input for new questions
            st.write("---")
            query = st.chat_input("Ask about the document...")
            if query:
                with st.spinner("Analyzing..."):
                    response = generate_rag_response(query, st.session_state.faiss_index, st.session_state.text_chunks)
                    st.session_state.chat_history.extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()  # Refresh to show new message
        else:
            st.info("Please upload and analyze a document first to use the Q&A feature.")

            # Sample Q&A to show functionality
            with st.expander("Sample Q&A"):
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <h4 style="margin-top: 0;">Example Questions You Can Ask:</h4>
                    <ul>
                        <li>What are the key obligations in this contract?</li>
                        <li>Explain the termination clause in simple terms.</li>
                        <li>What are the payment terms?</li>
                        <li>Are there any concerning liability clauses?</li>
                        <li>Summarize the confidentiality requirements.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ------------------------------ TAB 4: Document Comparison ------------------------------

    with tab4:
        st.header("Document Comparison")

        if st.session_state.document_processed:
            with st.container(border=True):
                compare_file = st.file_uploader("Upload Comparison Document", type=["pdf"])

                if compare_file:
                    try:
                        # Document Names
                        doc1_name = "Original Document"
                        doc2_name = compare_file.name

                        # Extract Text
                        compare_text = extract_text_from_pdf(compare_file)

                        # Generate Comparison
                        comparison = compare_documents(st.session_state.full_text, compare_text)
                        comparison_table = compare_documents_tabular(st.session_state.full_text, compare_text)
                        # Display Comparison Results
                        st.subheader("Visual Comparison")
                        st.markdown(
                            f'<div style="border:1px solid #eee; padding:20px; border-radius:8px">'
                            f'{comparison}</div>',
                            unsafe_allow_html=True
                        )
                        # Display Tabular Comparison
                        st.subheader("Tabular Comparison")
                        st.dataframe(comparison_table, use_container_width=True)

                        # Export Options
                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            if st.button("📊 Generate Detailed Comparison Report", use_container_width=True):
                                with st.spinner("Generating comparison report..."):
                                    report_buffer = export_comparison_report(
                                        st.session_state.full_text,
                                        compare_text,
                                        doc1_name,
                                        doc2_name
                                    )
                                    st.session_state.comparison_report = report_buffer
                                    st.success("Comparison report ready for download!")

                        with export_col2:
                            if st.session_state.get('comparison_report'):
                                st.download_button(
                                    label="⬇️ Download Comparison Report",
                                    data=st.session_state.comparison_report,
                                    file_name="Document_Comparison_Report.html",
                                    mime="text/html",
                                    use_container_width=True
                                )

                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")
                else:
                    st.info("Upload a second document to compare with your original document.")

        else:
            st.info("Please upload and analyze a document first before comparing.")

    # ------------------------------ TAB 5: Compliance ------------------------------

    with tab5:
        st.header("Document Compliance Requirements")

        if st.session_state.document_processed and hasattr(st.session_state, 'document_compliance'):
            compliance_data = st.session_state.document_compliance

            if not compliance_data:
                st.info("No specific compliance requirements identified for this document.")
            else:
                for category, data in compliance_data.items():
                    confidence = data.get('confidence', 0)
                    with st.expander(f"📋 {category} Compliance (Confidence: {confidence:.2f})"):  # Confidence Formatting
                        # Requirements section
                        st.subheader("Key Compliance Requirements")
                        for item in data.get('requirements', []):
                            st.markdown(f'<div class="compliance-item">{item}</div>', unsafe_allow_html=True)

                        # Relevant regulations section
                        st.subheader("Relevant Regulations")
                        regulations_html = ""
                        for regulation in data.get('relevant_regulations', []):
                            regulations_html += f'<span class="regulation-item">{regulation}</span>'
                        st.markdown(regulations_html, unsafe_allow_html=True)

                        # Recent updates section
                        updates = data.get('updates', [])
                        if updates:
                            st.subheader("Recent Regulatory Updates")
                            for update in updates:
                                st.markdown(f"""
                                <div class="update-card">
                                    <div class="update-title">{update.get('title', '')}</div>
                                    <div class="update-source">Source: {update.get('source', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze a document to see relevant compliance requirements.")

    # ------------------------------ TAB 6: Legal Updates ------------------------------

    with tab6:
        st.header("Document-Specific Legal Updates")

        if st.session_state.document_processed and hasattr(st.session_state, 'legal_updates'):
            legal_updates = st.session_state.legal_updates

            if not legal_updates:
                st.info("No relevant legal updates found for this document.")
            else:
                for category, data in legal_updates.items():
                    confidence = data['confidence']
                    with st.expander(f"📋 {category} Updates (Confidence: {confidence:.2f})"):  # Confidence Formatting
                        if not data['updates']:
                            st.info(f"No recent updates found for {category}")
                        else:
                            for update in data['updates']:
                                st.markdown(f"""
                                <div class="update-card">
                                    <div class="update-title">{update['title']}</div>
                                    <div class="update-source">Source: {update['source']}</div>
                                    <div class="update-snippet">{update.get('snippet', '')}</div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("Please upload and analyze a document to see relevant legal updates.")

    # ------------------------------ TAB 7: Report Generation ------------------------------

    with tab7:
        st.header("Report Generation")

        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Generate PDF Report")

                if st.button("📊 Generate Full Report PDF", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        pdf_buffer = generate_pdf(
                            st.session_state.summaries['document'],
                            st.session_state.risk_data,
                            st.session_state.legal_updates if hasattr(st.session_state, 'legal_updates') else None,
                            st.session_state.document_compliance if hasattr(st.session_state, 'document_compliance') else None
                        )
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("PDF ready for download!")

                if st.session_state.get('pdf_buffer'):  # Check if pdf_buffer exists
                    st.download_button(
                        label="⬇️ Download Full Report",
                        data=st.session_state.pdf_buffer,
                        file_name="Legal_Analysis_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            # Email Reports Section
            with st.container(border=True):
                st.subheader("Email Reports")
                email = st.text_input("Recipient Email Address", placeholder="legal@company.com")

                # Validate email
                if email and not validate_email(email):
                    st.error("Please enter a valid email address.")

                email_col1, email_col2 = st.columns(2)

                with email_col1:
                    if st.button("📧 Send Summary Report", use_container_width=True):
                        if not email or not validate_email(email):
                            st.warning("Please enter a valid email address.")
                        else:
                            with st.spinner("Sending summary report..."):
                                # Create a summary-only PDF
                                summary_pdf = generate_pdf(
                                    st.session_state.summaries['document'],
                                    None,  # No risk data
                                    None,  # No legal updates
                                    None   # No compliance data
                                )

                                # Create email content
                                email_html = create_email_text(
                                    summary=st.session_state.summaries['document']
                                )

                                # Send email
                                success, message = send_email(
                                    email,
                                    summary_pdf,
                                    "Your Legal Document Summary Report",
                                    email_html,
                                    "document_summary.pdf"
                                )

                                if success:
                                    st.success("Summary report sent successfully!")
                                else:
                                    st.error(message)

                with email_col2:
                    if st.button("📧 Send Complete Analysis", use_container_width=True):
                        if not email or not validate_email(email):
                            st.warning("Please enter a valid email address.")
                        elif not st.session_state.get('pdf_buffer'):
                            st.warning("Please generate the full report first.")
                        else:
                            with st.spinner("Sending complete analysis..."):
                                # Create email content with all sections
                                email_html = create_email_text(
                                    summary=st.session_state.summaries['document'],
                                    risk_assessment="Included in the attached PDF"
                                )

                                # Send email with the full report
                                success, message = send_email(
                                    email,
                                    st.session_state.pdf_buffer,
                                    "Your Complete Legal Document Analysis",
                                    email_html,
                                    "complete_legal_analysis.pdf"
                                )

                                if success:
                                    st.success("Complete analysis sent successfully!")
                                else:
                                    st.error(message)
        else:
            st.info("Please upload and analyze a document first to generate reports.")

if __name__ == "__main__":
    main()