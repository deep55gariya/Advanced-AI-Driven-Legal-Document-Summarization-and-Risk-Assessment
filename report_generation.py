from fpdf import FPDF
from io import BytesIO
import base64
import os
import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

from utils import get_sendgrid_credentials

def generate_pdf(summary, risk_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Legal Document Analysis Report", ln=True, align="C")
    pdf.ln(10)  # Line break

    pdf.multi_cell(0, 10, f"Document Summary:\n{summary}")
    pdf.ln(5)

    pdf.multi_cell(0, 10, f"Risk Score: {risk_data.get('total_score', 0)}")

    # Save PDF as a string
    pdf_data = pdf.output(dest="S").encode("latin1")  # Generate PDF as a string

    # Convert to BytesIO
    pdf_buffer = BytesIO(pdf_data)

    return pdf_buffer

def send_email(recipient_email, pdf_buffer):
    try:
        sendgrid_api_key, sender_email = get_sendgrid_credentials()
    except ValueError as e:
        st.error(f"⚠ {e}")
        return False

    # Read and encode PDF
    pdf_buffer.seek(0)
    encoded_pdf = base64.b64encode(pdf_buffer.read()).decode()

    # Construct email
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject="📄 Legal Document Report",
        html_content="Please find the attached legal document report."
    )

    # Attach PDF
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
        st.success("✅ Email sent successfully!")
        return True
    except Exception as e:
        st.error(f"⚠ Email sending failed: {str(e)}")
        return False