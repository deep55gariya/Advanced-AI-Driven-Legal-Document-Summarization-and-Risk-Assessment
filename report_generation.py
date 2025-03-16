from fpdf import FPDF
from io import BytesIO
import base64
import os
import streamlit as st
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from datetime import datetime

from utils import get_sendgrid_credentials

def generate_pdf(summary, risk_data, legal_updates=None, compliance_data=None):
    """Generate a PDF report with document analysis results"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts
    pdf.set_font("Arial", "B", 16)
    
    # Header
    pdf.cell(0, 10, "Legal Document Analysis Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
    pdf.ln(10)
    
    # Summary section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Document Summary", 0, 1)
    pdf.set_font("Arial", "", 11)
    
    # Replace Unicode bullet points with ASCII alternatives
    clean_summary = summary.replace("•", "*").replace("\u2022", "*")
    
    # Add summary text with word wrapping
    pdf.multi_cell(0, 6, clean_summary)
    pdf.ln(10)
    
    # Risk Assessment section
    if risk_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Risk Assessment", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        # Overall risk score
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Overall Risk Score: {risk_data.get('total_score', 'N/A')}/100", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        # Risk counts by severity
        pdf.cell(0, 8, "Risk Counts by Severity:", 0, 1)
        for severity, count in risk_data.get("severity_counts", {}).items():
            # Replace Unicode bullet points with ASCII alternatives
            pdf.cell(0, 6, f"* {severity}: {count}", 0, 1)
        
        # Risk categories
        if risk_data.get("categories"):
            pdf.ln(5)
            pdf.cell(0, 8, "Risk Categories:", 0, 1)
            for category, score in risk_data.get("categories", {}).items():
                # Replace Unicode bullet points with ASCII alternatives
                pdf.cell(0, 6, f"* {category}: {score}", 0, 1)
        
        pdf.ln(10)
    
    # Compliance section
    if compliance_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Compliance Requirements", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        for category, data in compliance_data.items():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"{category} Compliance", 0, 1)
            pdf.set_font("Arial", "", 11)
            
            # Requirements
            if data.get('requirements'):
                pdf.cell(0, 8, "Key Requirements:", 0, 1)
                for req in data.get('requirements', []):
                    # Replace Unicode bullet points with ASCII alternatives
                    clean_req = req.replace("•", "*").replace("\u2022", "*")
                    pdf.multi_cell(0, 6, f"* {clean_req}")
            
            # Regulations
            if data.get('relevant_regulations'):
                pdf.ln(3)
                pdf.cell(0, 8, "Relevant Regulations:", 0, 1)
                for reg in data.get('relevant_regulations', []):
                    # Replace Unicode bullet points with ASCII alternatives
                    clean_reg = reg.replace("•", "*").replace("\u2022", "*")
                    pdf.multi_cell(0, 6, f"* {clean_reg}")
            
            pdf.ln(5)
        
        pdf.ln(5)
    
    # Legal Updates section
    if legal_updates:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recent Legal Updates", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        for category, data in legal_updates.items():
            if data.get('updates'):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, f"{category} Updates", 0, 1)
                pdf.set_font("Arial", "", 11)
                
                for update in data.get('updates', []):
                    # Replace Unicode bullet points with ASCII alternatives
                    clean_title = update.get('title', '').replace("•", "*").replace("\u2022", "*")
                    clean_source = update.get('source', '').replace("•", "*").replace("\u2022", "*")
                    
                    pdf.set_font("Arial", "B", 11)
                    pdf.multi_cell(0, 6, f"* {clean_title}")
                    pdf.set_font("Arial", "", 10)
                    pdf.multi_cell(0, 6, f"  Source: {clean_source}")
                    pdf.ln(3)
                
                pdf.ln(5)
    
    # Try to generate PDF with error handling
    try:
        pdf_data = pdf.output(dest="S").encode("latin1")  # Generate PDF as a string
        return BytesIO(pdf_data)
    except UnicodeEncodeError:
        # If encoding fails, try a more aggressive character replacement approach
        try:
            # Create a new PDF with even more aggressive character replacement
            return generate_pdf_with_ascii_only(summary, risk_data, legal_updates, compliance_data)
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}")
            return BytesIO(b"Error generating PDF report")

def generate_pdf_with_ascii_only(summary, risk_data, legal_updates=None, compliance_data=None):
    """Fallback PDF generator that strictly uses ASCII characters only"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set up fonts
    pdf.set_font("Arial", "B", 16)
    
    # Header
    pdf.cell(0, 10, "Legal Document Analysis Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
    pdf.ln(10)
    
    # Function to sanitize text for latin-1 encoding
    def sanitize_text(text):
        if not isinstance(text, str):
            return str(text)
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '\u2022': '-',  # bullet point
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2013': '-',  # en dash
            '\u2014': '--', # em dash
            '\u2026': '...' # ellipsis
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        # Remove any remaining non-latin1 characters
        return ''.join(c for c in text if ord(c) < 256)
    
    # Summary section with sanitized text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Document Summary", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, sanitize_text(summary))
    pdf.ln(10)
    
    # Risk Assessment section
    if risk_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Risk Assessment", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        # Overall risk score
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Overall Risk Score: {risk_data.get('total_score', 'N/A')}/100", 0, 1)
        pdf.set_font("Arial", "", 11)
        
        # Risk counts by severity
        pdf.cell(0, 8, "Risk Counts by Severity:", 0, 1)
        for severity, count in risk_data.get("severity_counts", {}).items():
            pdf.cell(0, 6, f"- {sanitize_text(severity)}: {count}", 0, 1)
        
        # Risk categories
        if risk_data.get("categories"):
            pdf.ln(5)
            pdf.cell(0, 8, "Risk Categories:", 0, 1)
            for category, score in risk_data.get("categories", {}).items():
                pdf.cell(0, 6, f"- {sanitize_text(category)}: {score}", 0, 1)
        
        pdf.ln(10)
    
    # Compliance section with sanitized text
    if compliance_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Compliance Requirements", 0, 1)
        
        for category, data in compliance_data.items():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"{sanitize_text(category)} Compliance", 0, 1)
            pdf.set_font("Arial", "", 11)
            
            # Requirements
            if data.get('requirements'):
                pdf.cell(0, 8, "Key Requirements:", 0, 1)
                for req in data.get('requirements', []):
                    pdf.multi_cell(0, 6, f"- {sanitize_text(req)}")
            
            # Regulations
            if data.get('relevant_regulations'):
                pdf.ln(3)
                pdf.cell(0, 8, "Relevant Regulations:", 0, 1)
                for reg in data.get('relevant_regulations', []):
                    pdf.multi_cell(0, 6, f"- {sanitize_text(reg)}")
            
            pdf.ln(5)
    
    # Legal Updates section with sanitized text
    if legal_updates:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recent Legal Updates", 0, 1)
        
        for category, data in legal_updates.items():
            if data.get('updates'):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, f"{sanitize_text(category)} Updates", 0, 1)
                pdf.set_font("Arial", "", 11)
                
                for update in data.get('updates', []):
                    pdf.set_font("Arial", "B", 11)
                    pdf.multi_cell(0, 6, f"- {sanitize_text(update.get('title', ''))}")
                    pdf.set_font("Arial", "", 10)
                    pdf.multi_cell(0, 6, f"  Source: {sanitize_text(update.get('source', ''))}")
                    pdf.ln(3)
                
                pdf.ln(5)
    
    # Generate PDF
    pdf_data = pdf.output(dest="S").encode("latin1")
    return BytesIO(pdf_data)

def send_email(recipient_email, attachment=None, subject=None, body=None, attachment_name=None):
    """
    Send an email with optional attachment using SendGrid
    Returns a tuple of (success_boolean, message_string)
    """
    try:
        sendgrid_api_key, sender_email = get_sendgrid_credentials()
    except ValueError as e:
        return False, f"⚠ {e}"

    # Default values if not provided
    if subject is None:
        subject = "📄 Legal Document Report"
    
    if body is None:
        body = """
        <h2>Legal Document Analysis Report</h2>
        <p>Please find attached your comprehensive legal document analysis report, which includes:</p>
        <ul>
            <li>Document summary</li>
            <li>Risk analysis</li>
            <li>Compliance requirements</li>
            <li>Relevant legal updates</li>
        </ul>
        <p>Thank you for using our service.</p>
        """
    
    # Create message
    message = Mail(
        from_email=sender_email,
        to_emails=recipient_email,
        subject=subject,
        html_content=body
    )
    
    # Add attachment if provided
    if attachment:
        attachment.seek(0)
        pdf_data = attachment.read()
        encoded_pdf = base64.b64encode(pdf_data).decode()
        
        file_attachment = Attachment(
            FileContent(encoded_pdf),
            FileName(attachment_name or "Legal_Report.pdf"),
            FileType("application/pdf"),
            Disposition("attachment")
        )
        message.attachment = file_attachment

    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        return True, f"Email sent successfully. Status code: {response.status_code}"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

def create_email_text(summary=None, risk_assessment=None):
    """Create HTML email content based on available analysis components"""
    email_html = """
    <h2>Legal Document Analysis Report</h2>
    <p>Dear User,</p>
    <p>Please find attached the analysis of your uploaded legal document.</p>
    """
    
    if summary:
        email_html += "<h3>Document Summary</h3>"
        email_html += f"<p>A summary of your document has been included in the attached PDF.</p>"
    
    if risk_assessment:
        email_html += "<h3>Risk Assessment</h3>"
        email_html += f"<p>A comprehensive risk assessment has been included in the attached PDF.</p>"
    
    email_html += """
    <p>This report was generated by the AI-Driven Legal Document Analysis System.</p>
    <p>Thank you for using our service.</p>
    """
    
    return email_html