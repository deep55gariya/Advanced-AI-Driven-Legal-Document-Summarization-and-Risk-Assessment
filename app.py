
# best
import streamlit as st
import cohere
import fitz  
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from dotenv import load_dotenv
import requests
import datetime
import base64

load_dotenv()

# Initialize Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)


st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

# Sidebar - File Uploader
st.sidebar.title("📂 Upload Legal Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Email Sending Function
def send_email(recipient_email, pdf_buffer):
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL")
    pdf_buffer.seek(0)
    pdf_data = pdf_buffer.read()
    encoded_pdf = base64.b64encode(pdf_data).decode()
    
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject="📄 Legal Document Report",
        html_content="Please find the attached legal document report."
    )
    
    attachment = Attachment(
        FileContent(encoded_pdf),
        FileName("Legal_Report.pdf"),
        FileType("application/pdf"),
        Disposition("attachment")
    )
    message.attachment = attachment
    
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        sg.send(message)
        return True
    except Exception as e:
        st.error(f"⚠ Email sending failed: {e}")
        return False
    
# Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    return text

# Summarization Function
def summarize_document(text):
    chunks = text[:4000]
    prompt = f"Summarize the following legal document:\n\n{chunks}"
    response = co.generate(prompt=prompt, model="command", max_tokens=300)
    return response.generations[0].text.strip()

# Risk Identification Function
def identify_risks(text):
    chunks = text[:4000]
    prompt = f"Identify potential legal risks in the following document:\n\n{chunks}"
    response = co.generate(prompt=prompt, model="command", max_tokens=300)
    return response.generations[0].text.strip()


import matplotlib.pyplot as plt

import PyPDF2


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to chunk text into smaller parts
def chunk_text(text, max_tokens=4000):
    return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]

# Function to identify risks in the document
def identify_risks(text):
    chunks = chunk_text(text)
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    all_risks = [] 

    for chunk in chunks:
        prompt = f"""
        Identify all potential risks in the following document and categorize them into Low, Medium, or High risk:
        \n\n{chunk}
        \n\nReturn the risks in this format: Risk Name - Risk Level (Low/Medium/High)
        """
        response = co.generate(prompt=prompt, model="command", max_tokens=500)

        risks = response.generations[0].text.strip().split("\n")
        all_risks.extend(risks)

        for risk in risks:
            if "High" in risk:
                risk_counts["High"] += 1
            elif "Medium" in risk:
                risk_counts["Medium"] += 1
            elif "Low" in risk:
                risk_counts["Low"] += 1

    return all_risks, risk_counts

# Function to plot risk distribution graph
def plot_risk_chart(risk_data):
    labels = risk_data.keys()
    values = risk_data.values()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=['green', 'orange', 'red'])
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Risks")
    ax.set_title("Risk Distribution in Document")
    st.pyplot(fig)

# Answering Function
def answer_question(text, question):
    chunks = text[:4000]
    prompt = f"Given the following legal document:\n\n{chunks}\n\nAnswer the question:\n{question}"
    response = co.chat(message=prompt, model="command-r")
    return response.text.strip()

import requests
from bs4 import BeautifulSoup
import streamlit as st

def fetch_gdpr_rules():
    """Scrape GDPR compliance rules from gdpr.eu website."""
    try:
        url = "https://gdpr.eu/compliance-checklist/"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        rules = [item.get_text(strip=True) for item in soup.find_all("li") if len(item.get_text(strip=True)) > 10]
        return rules

    except requests.RequestException:
        return []

def check_gdpr_compliance(file_content):
    """Check if uploaded file violates GDPR rules and return YES/NO."""
    gdpr_rules = fetch_gdpr_rules()
    if not gdpr_rules:
        return "Error fetching GDPR rules."

    for rule in gdpr_rules:
        if rule.lower() in file_content.lower():
            return "YES"  # Violates GDPR

    return "NO"  # Does not violate GDPR


# PDF Report Generation
def generate_pdf(summary, risks, chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    elements.append(Paragraph("📄 Legal Document Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("📄 Summary", styles["Heading2"]))
    elements.append(Paragraph(summary.replace("\n", "<br/>"), styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("⚠ Risks Identified", styles["Heading2"]))
    elements.append(Paragraph("\n".join(risks).replace("\n", "<br/>"), styles["Normal"]))

    elements.append(Spacer(1, 12))

    if chat_history:
        elements.append(Paragraph("💬 Chat History", styles["Heading2"]))
        for chat in chat_history:
            question, answer = chat
            elements.append(Paragraph(f"<b>You:</b> {question}", styles["Normal"]))
            elements.append(Paragraph(f"<b>AI:</b> {answer}", styles["Normal"]))
            elements.append(Spacer(1, 6))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# UI - Tabs Section
if uploaded_file:
    document_text = extract_text_from_pdf(uploaded_file)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 PDF Analysis", "⚠️ Risk Detection", "🤖 Chatbot", "📄 Report & Email", "📜 GDPR Compliance Checklist"])

    with tab1:
        st.subheader("📄 Summary")
        if "summary" not in st.session_state:
            st.session_state.summary = summarize_document(document_text)
        st.info(st.session_state.summary)

    with tab2:
        st.subheader("⚠ Risk Analysis")

        if "risks" not in st.session_state:
            st.session_state.risks = None
            st.session_state.risk_data = None

        if uploaded_file:
            document_text = extract_text_from_pdf(uploaded_file)
            st.success("✅ PDF uploaded and text extracted!")

            if st.button("Analyze Risks"):
                with st.spinner("Identifying risks..."):
                    risks, risk_data = identify_risks(document_text)
                    st.session_state.risks = risks  # Store results in session state
                    st.session_state.risk_data = risk_data

        # Show identified risks if available
        if st.session_state.risks:
            st.subheader("Identified Risks")
            for risk in st.session_state.risks:
                st.warning(risk)

        # Show risk graph if data available
        if st.session_state.risk_data:
            st.subheader("📊 Risk Distribution Graph")
            plot_risk_chart(st.session_state.risk_data)


    with tab3:
        st.subheader("💬 AI Chatbot")
        user_question = st.text_input("Ask a question about the document:")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if st.button("Get Answer"):
            if user_question.strip() == "":
                st.warning("⚠ Please enter a valid question!")
            else:
                with st.spinner("Analyzing... 🔍"):
                    answer = answer_question(document_text, user_question)
                    st.session_state.chat_history.append((user_question, answer))

        # Display full chat history
        for chat in st.session_state.chat_history:
            question, answer = chat
            st.markdown(f"""
            <div style="text-align: left; background-color: #f0f0f5; padding: 10px; border-radius: 10px; margin-bottom: 5px; width: 70%;">
                <b>You:</b> {question}
            </div>
            <div style="text-align: right; background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin-bottom: 5px; width: 70%; margin-left: 30%;">
                <b>AI:</b> {answer}
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.subheader("📥 Download & Email Report")

        save_chat = st.checkbox("Include Chat History in PDF", value=True)

        if st.button("Generate & Download Report"):
            chat_history_to_save = st.session_state.chat_history if save_chat else []
            pdf_buffer = generate_pdf(st.session_state.summary, st.session_state.risks, chat_history_to_save)
            st.download_button("📥 Download Report", pdf_buffer, "Legal_Report.pdf", "application/pdf")

        recipient_email = st.text_input("📧 Enter recipient's email:")
        if st.button("Send Report via Email"):
            if recipient_email.strip() == "":
                st.warning("⚠ Please enter a valid email address!")
            else:
                chat_history_to_save = st.session_state.chat_history if save_chat else []
                pdf_buffer = generate_pdf(st.session_state.summary, st.session_state.risks, chat_history_to_save)
                if send_email(recipient_email, pdf_buffer):
                    st.success("✅ Email sent successfully!")
    
    with tab5:
        st.subheader("GDPR Compliance Checklist")
        uploaded_file = st.file_uploader("Upload a document to check GDPR compliance", type=["txt", "pdf", "docx"])
    

        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            compliance_result = check_gdpr_compliance(file_content)

            if compliance_result == "YES":
                st.error("❌ Your document VIOLATES GDPR rules.")
            elif compliance_result == "NO":
                st.success("✅ Your document is GDPR COMPLIANT.")
            else:
                st.warning(compliance_result)
