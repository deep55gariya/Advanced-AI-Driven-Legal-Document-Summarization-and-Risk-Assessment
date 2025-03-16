import difflib
import streamlit as st
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import datetime
from io import BytesIO
import pandas as pd

# Try to load the model, with graceful fallback if not available
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.warning(f"Could not load semantic comparison model: {str(e)}. Falling back to text-only comparison.")
        return None

# Initialize embedding model
embedding_model = load_embedding_model()

def preprocess_text(text: str) -> str:
    """Preprocess text to handle common formatting issues in legal documents"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Standardize newlines
    text = re.sub(r'(\r\n|\r|\n)', '\n', text)
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()

def extract_document_sections(text: str) -> Dict[str, str]:
    """Attempt to extract common sections from legal documents"""
    sections = {}
    
    # Common legal document section patterns
    section_patterns = [
        (r'(?i)(?:^|\n\s*)(PREAMBLE|RECITALS)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Preamble"),
        (r'(?i)(?:^|\n\s*)(WHEREAS)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Recitals"),
        (r'(?i)(?:^|\n\s*)(DEFINITIONS)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Definitions"),
        (r'(?i)(?:^|\n\s*)(OBLIGATIONS|RESPONSIBILITIES)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Obligations"),
        (r'(?i)(?:^|\n\s*)(PAYMENT\s+TERMS?|COMPENSATION)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Payment"),
        (r'(?i)(?:^|\n\s*)(TERMINATION)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Termination"),
        (r'(?i)(?:^|\n\s*)(CONFIDENTIALITY)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Confidentiality"),
        (r'(?i)(?:^|\n\s*)(GOVERNING\s+LAW|JURISDICTION)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Governing Law"),
        (r'(?i)(?:^|\n\s*)(INDEMNIFICATION|INDEMNITY)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Indemnification"),
        (r'(?i)(?:^|\n\s*)(FORCE\s+MAJEURE)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Force Majeure"),
        (r'(?i)(?:^|\n\s*)(NOTICES?)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Notices"),
        (r'(?i)(?:^|\n\s*)(ASSIGNMENT)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Assignment"),
        (r'(?i)(?:^|\n\s*)(ENTIRE\s+AGREEMENT)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Entire Agreement"),
        (r'(?i)(?:^|\n\s*)(AMENDMENTS?|MODIFICATIONS?)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Amendments"),
        (r'(?i)(?:^|\n\s*)(SIGNATURES?|EXECUTION)[:\s]*(.*?)(?=\n\s*[A-Z][A-Z\s]+[:\s]|\Z)', "Signatures"),
    ]
    
    for pattern, name in section_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for _, content in matches:
                sections[name] = content.strip()
    
    return sections

def compare_semantic_similarity(text1: str, text2: str) -> float:
    """Compare the semantic similarity between two texts using sentence embeddings"""
    if embedding_model is None:
        return 0.0  # Fall back if model not available
    
    try:
        embedding1 = embedding_model.encode(text1, convert_to_numpy=True)
        embedding2 = embedding_model.encode(text2, convert_to_numpy=True)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    except Exception as e:
        st.error(f"Error calculating semantic similarity: {str(e)}")
        return 0.0

def extract_key_terms(text: str, top_n: int = 10) -> List[str]:
    """Extract key legal terms from the document"""
    # Common legal terms to look for
    legal_terms = [
        "agreement", "contract", "party", "parties", "terms", "conditions", "obligations",
        "rights", "law", "jurisdiction", "liability", "confidential", "termination",
        "payment", "damages", "breach", "remedy", "warranty", "representation",
        "indemnify", "comply", "dispute", "arbitration", "governing", "severability"
    ]
    
    # Tokenize and count terms
    words = word_tokenize(text.lower())
    counts = Counter(words)
    
    # Get top legal terms
    top_terms = []
    for term in legal_terms:
        if term in counts:
            top_terms.append((term, counts[term]))
    
    # Sort by frequency and get top N
    top_terms.sort(key=lambda x: x[1], reverse=True)
    return [term for term, count in top_terms[:top_n]]

def calculate_statistics(text1: str, text2: str) -> Dict:
    """Calculate comparison statistics between two documents"""
    stats = {}
    
    # Word counts
    words1 = word_tokenize(text1)
    words2 = word_tokenize(text2)
    stats["word_count_doc1"] = len(words1)
    stats["word_count_doc2"] = len(words2)
    stats["word_diff_percentage"] = round(abs(len(words1) - len(words2)) / max(len(words1), len(words2)) * 100, 2)
    
    # Sentence counts
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    stats["sentence_count_doc1"] = len(sentences1)
    stats["sentence_count_doc2"] = len(sentences2)
    
    # Semantic similarity if model is available
    if embedding_model is not None:
        stats["semantic_similarity"] = round(compare_semantic_similarity(text1, text2) * 100, 2)
    
    # Number of differences
    differ = difflib.Differ()
    diff = list(differ.compare(text1.splitlines(), text2.splitlines()))
    stats["added_lines"] = len([line for line in diff if line.startswith('+ ')])
    stats["removed_lines"] = len([line for line in diff if line.startswith('- ')])
    stats["changed_lines"] = len([line for line in diff if line.startswith('? ')])
    
    return stats

def generate_diff_summary(diff_lines: List[str]) -> List[str]:
    """Generate a summary of key differences from diff lines"""
    summary = []
    
    current_section = None
    section_diffs = {}
    
    for line in diff_lines:
        if line.startswith('- ') or line.startswith('+ '):
            # Check if line might be a section header
            cleaned_line = line[2:].strip().upper()
            
            # Look for potential section headers
            if re.match(r'^[A-Z][A-Z\s]+[:.]\s*$', cleaned_line):
                current_section = cleaned_line
                if current_section not in section_diffs:
                    section_diffs[current_section] = []
            
            # Add diff to current section
            if current_section:
                section_diffs[current_section].append(line)
            else:
                # General differences
                if "General" not in section_diffs:
                    section_diffs["General"] = []
                section_diffs["General"].append(line)
    
    # Create summary
    for section, diffs in section_diffs.items():
        if len(diffs) > 0:
            summary.append(f"Changes in section {section}:")
            # Limit to first 3 differences per section to keep summary concise
            for diff in diffs[:3]:
                if diff.startswith('- '):
                    summary.append(f"  • Removed: {diff[2:].strip()[:100]}...")
                elif diff.startswith('+ '):
                    summary.append(f"  • Added: {diff[2:].strip()[:100]}...")
            
            if len(diffs) > 3:
                summary.append(f"  • ... and {len(diffs) - 3} more changes")
    
    return summary

def compare_documents(doc1: str, doc2: str) -> str:
    """Enhanced document comparison with structural analysis and visualization"""
    try:
        # Preprocess texts
        doc1_cleaned = preprocess_text(doc1)
        doc2_cleaned = preprocess_text(doc2)
        
        # Extract document sections for structured comparison
        doc1_sections = extract_document_sections(doc1_cleaned)
        doc2_sections = extract_document_sections(doc2_cleaned)
        
        # Calculate statistics
        stats = calculate_statistics(doc1_cleaned, doc2_cleaned)
        
        # Extract key terms
        doc1_terms = extract_key_terms(doc1_cleaned)
        doc2_terms = extract_key_terms(doc2_cleaned)
        
        # Generate standard diff
        d = difflib.Differ()
        diff = list(d.compare(doc1_cleaned.splitlines(), doc2_cleaned.splitlines()))
        
        # Generate diff summary
        diff_summary = generate_diff_summary(diff)
        
        # Build the comparison HTML output
        html_output = []
        
        # Add statistics section
        html_output.append('<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">')
        html_output.append('<h3 style="margin-top: 0;">Document Comparison Summary</h3>')
        
        # Statistics table
        html_output.append('<table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">')
        html_output.append('<tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Metric</th><th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Document 1</th><th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Document 2</th><th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Difference</th></tr>')
        
        # Word count row
        html_output.append(f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Word Count</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{stats["word_count_doc1"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{stats["word_count_doc2"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{abs(stats["word_count_doc1"] - stats["word_count_doc2"])} ({stats["word_diff_percentage"]}%)</td></tr>')
        
        # Sentence count row
        html_output.append(f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Sentence Count</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{stats["sentence_count_doc1"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{stats["sentence_count_doc2"]}</td><td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{abs(stats["sentence_count_doc1"] - stats["sentence_count_doc2"])}</td></tr>')
        
        # Changes row
        html_output.append(f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;">Changes</td><td colspan="3" style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{stats["added_lines"]} additions, {stats["removed_lines"]} removals</td></tr>')
        
        # Semantic similarity if available
        if "semantic_similarity" in stats:
            similarity_color = "#4CAF50" if stats["semantic_similarity"] > 80 else "#FFC107" if stats["semantic_similarity"] > 50 else "#F44336"
            html_output.append(f'<tr><td style="padding: 8px;">Semantic Similarity</td><td colspan="3" style="text-align: right; padding: 8px;"><span style="color: {similarity_color}; font-weight: bold;">{stats["semantic_similarity"]}%</span></td></tr>')
        
        html_output.append('</table>')
        
        # Add key differences summary
        if diff_summary:
            html_output.append('<div style="margin-bottom: 15px;">')
            html_output.append('<h4 style="margin-top: 0;">Key Differences</h4>')
            html_output.append('<ul style="margin-top: 5px; padding-left: 20px;">')
            for item in diff_summary:
                if item.startswith('Changes in section'):
                    if not item.endswith(':'):
                        html_output.append(f'</ul><h5 style="margin-bottom: 5px;">{item}</h5><ul style="margin-top: 5px; padding-left: 20px;">')
                else:
                    html_output.append(f'<li>{item}</li>')
            html_output.append('</ul>')
            html_output.append('</div>')
        
        html_output.append('</div>')
        
        # Section comparison
        if doc1_sections or doc2_sections:
            html_output.append('<div style="margin-bottom: 20px;">')
            html_output.append('<h3>Section Comparison</h3>')
            
            # Find all unique section names
            all_sections = set(list(doc1_sections.keys()) + list(doc2_sections.keys()))
            
            # Compare each section
            for section in sorted(all_sections):
                html_output.append(f'<h4>{section}</h4>')
                
                if section in doc1_sections and section in doc2_sections:
                    # If section is in both documents, show diff
                    if doc1_sections[section] == doc2_sections[section]:
                        html_output.append('<p style="color: #4CAF50;">✓ Identical in both documents</p>')
                    else:
                        # Calculate similarity
                        sim = compare_semantic_similarity(doc1_sections[section], doc2_sections[section])
                        sim_percentage = round(sim * 100, 1)
                        sim_color = "#4CAF50" if sim_percentage > 80 else "#FFC107" if sim_percentage > 50 else "#F44336"
                        
                        html_output.append(f'<p>Similarity: <span style="color: {sim_color}; font-weight: bold;">{sim_percentage}%</span></p>')
                        
                        # Generate a diff just for this section
                        section_diff = list(d.compare(
                            doc1_sections[section].splitlines(),
                            doc2_sections[section].splitlines()
                        ))
                        
                        html_output.append('<pre style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-size: 12px;">')
                        for line in section_diff:
                            if line.startswith('+ '):
                                html_output.append(f'<span style="color:green">{line}</span><br>')
                            elif line.startswith('- '):
                                html_output.append(f'<span style="color:red">{line}</span><br>')
                            elif line.startswith('? '):
                                html_output.append(f'<span style="color:gray">{line}</span><br>')
                            else:
                                html_output.append(f'{line}<br>')
                        html_output.append('</pre>')
                
                elif section in doc1_sections:
                    html_output.append('<p style="color: #F44336;">❌ Section removed in Document 2</p>')
                    # Show the section from doc1
                    html_output.append('<pre style="background-color: #ffebee; padding: 10px; border-radius: 5px; max-height: 150px; overflow-y: auto; font-size: 12px;">')
                    html_output.append(doc1_sections[section][:500] + ('...' if len(doc1_sections[section]) > 500 else ''))
                    html_output.append('</pre>')
                
                else:  # Section in doc2 only
                    html_output.append('<p style="color: #4CAF50;">✅ Section added in Document 2</p>')
                    # Show the section from doc2
                    html_output.append('<pre style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; max-height: 150px; overflow-y: auto; font-size: 12px;">')
                    html_output.append(doc2_sections[section][:500] + ('...' if len(doc2_sections[section]) > 500 else ''))
                    html_output.append('</pre>')
            
            html_output.append('</div>')
        
        # Full diff visualization
        html_output.append('<h3>Full Document Comparison</h3>')
        html_output.append('<pre style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-size: 12px; line-height: 1.5;">')
        for line in diff:
            if line.startswith('+ '):
                html_output.append(f'<span style="color:green">{line}</span><br>')
            elif line.startswith('- '):
                html_output.append(f'<span style="color:red">{line}</span><br>')
            elif line.startswith('? '):
                continue  # Skip the markers to reduce clutter
            else:
                html_output.append(f'{line}<br>')
        html_output.append('</pre>')
        
        return ''.join(html_output)
        
    except Exception as e:
        return f"Comparison failed: {str(e)}"

def compare_documents_tabular(doc1: str, doc2: str) -> pd.DataFrame:
    """
    Create a tabular comparison between two documents, showing section-by-section differences
    in a structured format.
    """
    try:
        # Preprocess texts
        doc1_cleaned = preprocess_text(doc1)
        doc2_cleaned = preprocess_text(doc2)
        
        # Extract document sections for structured comparison
        doc1_sections = extract_document_sections(doc1_cleaned)
        doc2_sections = extract_document_sections(doc2_cleaned)
        
        # Find all unique section names
        all_sections = set(list(doc1_sections.keys()) + list(doc2_sections.keys()))
        
        # Prepare data for DataFrame
        comparison_data = []
        
        # First add a row for overall document comparison
        doc1_word_count = len(word_tokenize(doc1_cleaned))
        doc2_word_count = len(word_tokenize(doc2_cleaned))
        overall_similarity = compare_semantic_similarity(doc1_cleaned, doc2_cleaned)
        
        comparison_data.append({
            "Section": "OVERALL DOCUMENT",
            "Present in Doc1": "Yes",
            "Present in Doc2": "Yes",
            "Doc1 Length": doc1_word_count,
            "Doc2 Length": doc2_word_count,
            "Length Diff": doc2_word_count - doc1_word_count,
            "Similarity": f"{overall_similarity*100:.1f}%",
            "Status": "Modified" if overall_similarity < 0.98 else "Identical"
        })
        
        # Add rows for each section
        for section in sorted(all_sections):
            in_doc1 = section in doc1_sections
            in_doc2 = section in doc2_sections
            
            row_data = {
                "Section": section,
                "Present in Doc1": "Yes" if in_doc1 else "No",
                "Present in Doc2": "Yes" if in_doc2 else "No",
                "Doc1 Length": len(word_tokenize(doc1_sections.get(section, ""))) if in_doc1 else 0,
                "Doc2 Length": len(word_tokenize(doc2_sections.get(section, ""))) if in_doc2 else 0,
            }
            
            # Calculate length difference
            row_data["Length Diff"] = row_data["Doc2 Length"] - row_data["Doc1 Length"]
            
            # Determine status and similarity
            if in_doc1 and in_doc2:
                similarity = compare_semantic_similarity(doc1_sections[section], doc2_sections[section])
                row_data["Similarity"] = f"{similarity*100:.1f}%"
                
                if similarity > 0.98:  # Almost identical
                    row_data["Status"] = "Identical"
                elif similarity > 0.8:  # Minor changes
                    row_data["Status"] = "Minor Changes"
                elif similarity > 0.5:  # Significant changes
                    row_data["Status"] = "Major Changes"
                else:  # Completely different
                    row_data["Status"] = "Rewritten"
            elif in_doc1:
                row_data["Similarity"] = "0%"
                row_data["Status"] = "Removed"
            else:  # in_doc2 only
                row_data["Similarity"] = "0%"
                row_data["Status"] = "Added"
            
            comparison_data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Style the DataFrame for better visualization
        return df
    
    except Exception as e:
        st.error(f"Tabular comparison failed: {str(e)}")
        # Return empty DataFrame in case of error
        return pd.DataFrame()

def export_comparison_report(doc1: str, doc2: str, doc1_name: str = "Document 1", doc2_name: str = "Document 2") -> BytesIO:
    """Export the comparison as a standalone HTML report"""
    
    try:
        comparison_html = compare_documents(doc1, doc2)
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document Comparison: {doc1_name} vs {doc2_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3, h4, h5 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ padding: 20px 0; border-bottom: 1px solid #eee; margin-bottom: 20px; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; color: #777; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Document Comparison Report</h1>
                    <p><strong>{doc1_name}</strong> vs <strong>{doc2_name}</strong></p>
                    <p>Generated on: {timestamp}</p>
                </div>
                
                {comparison_html}
                
                <div class="footer">
                    <p>This report was generated automatically by LegalDoc Analyst.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Convert to bytes for download
        bytes_io = BytesIO(full_html.encode())
        return bytes_io
        
    except Exception as e:
        st.error(f"Failed to export comparison report: {str(e)}")
        return BytesIO(b"Error generating report")
