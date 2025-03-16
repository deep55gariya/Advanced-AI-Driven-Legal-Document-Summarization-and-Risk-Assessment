import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain_groq import ChatGroq
import os
import time
import re
from typing import Dict, List, Tuple, Optional

# URLs for regulatory and legal updates
LEGAL_UPDATE_SOURCES = {
    "GDPR": [
        "https://gdpr-info.eu/",
        "https://ec.europa.eu/newsroom/just/items/612053",
    ],
    "HIPAA": [
        "https://www.hhs.gov/hipaa/index.html",
        "https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/index.html",
    ],
    "Contracts": [
        "https://www.nolo.com/legal-updates/business-law",
        "https://www.americanbar.org/groups/business_law/publications/",
    ],
    "Intellectual Property": [
        "https://www.wipo.int/portal/en/news.html",
        "https://www.uspto.gov/about-us/news-updates",
    ],
    "Employment Law": [
        "https://www.eeoc.gov/newsroom",
        "https://www.dol.gov/newsroom/releases",
    ],
    "Real Estate": [
        "https://www.nar.realtor/legal",
        "https://www.hud.gov/press",
    ],
    "Tax Law": [
        "https://www.irs.gov/newsroom",
        "https://www.taxnotes.com/tax-notes-today-federal",
    ],
}

def classify_document_type(text: str) -> List[Tuple[str, float]]:
    """
    Classify the document to determine the most likely legal categories.
    Returns list of (category, confidence) pairs sorted by confidence.
    """
    # Initialize keyword patterns for different document types
    keywords = {
        "GDPR": r"\b(GDPR|General\s+Data\s+Protection\s+Regulation|data\s+protection|personal\s+data|data\s+subject|controller|processor)\b",
        "HIPAA": r"\b(HIPAA|Health\s+Insurance\s+Portability|PHI|covered\s+entity|protected\s+health\s+information)\b",
        "Contracts": r"\b(contract|agreement|parties|consideration|clause|termination|breach|obligations)\b",
        "Intellectual Property": r"\b(patent|trademark|copyright|intellectual\s+property|IP|invention|license|royalty)\b",
        "Employment Law": r"\b(employee|employer|employment|compensation|salary|benefit|termination|workplace|severance)\b",
        "Real Estate": r"\b(property|lease|tenant|landlord|real\s+estate|mortgage|rent|premises)\b",
        "Tax Law": r"\b(tax|taxation|deduction|income|exemption|IRS|audit|revenue)\b",
    }
    
    # Calculate frequencies
    results = []
    text_lower = text.lower()
    
    for category, pattern in keywords.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Simple scoring based on keyword frequency
        frequency = len(matches) / max(1, len(text_lower.split()))
        # Adjust scoring based on domain weightings
        results.append((category, frequency * 1000))  # Scaling for readability
    
    # Sort by confidence (highest first)
    return sorted(results, key=lambda x: x[1], reverse=True)

def clean_text(text: str) -> str:
    """Clean scraped text for better readability"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000] + '...' if len(text) > 1000 else text

def fetch_with_retry(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Fetch URL with retry logic and rotation of user agents"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
    
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': user_agents[attempt % len(user_agents)]}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            
            # If rate limited or forbidden, wait longer
            if response.status_code in [403, 429]:
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception as e:
            st.error(f"Request failed (attempt {attempt+1}): {str(e)}")
            time.sleep(1)
    
    return None

def fetch_updates_for_document(document_text: str) -> Dict:
    """
    Based on document classification, fetch relevant legal updates
    """
    document_categories = classify_document_type(document_text)
    
    # Take top 2 most likely categories
    top_categories = document_categories[:2]
    
    updates = {}
    for category, confidence in top_categories:
        if confidence < 1.0:  # Skip if confidence is very low
            continue
            
        updates[category] = {
            "confidence": round(confidence, 2),
            "updates": []
        }
        
        # For each category, fetch from all sources
        if category in LEGAL_UPDATE_SOURCES:
            for source_url in LEGAL_UPDATE_SOURCES[category]:
                try:
                    response = fetch_with_retry(source_url)
                    if not response:
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract headlines and update information
                    # Different parsing strategies based on URL patterns
                    if "gdpr-info" in source_url:
                        news_items = soup.select('article h2 a, article p')
                        for i in range(0, min(10, len(news_items)), 2):
                            if i+1 < len(news_items):
                                title = news_items[i].get_text(strip=True)
                                snippet = news_items[i+1].get_text(strip=True)
                                updates[category]["updates"].append({
                                    "title": title,
                                    "snippet": clean_text(snippet),
                                    "source": source_url
                                })
                    elif "hhs.gov" in source_url:
                        news_items = soup.select('.news-title, .field-content')
                        for item in news_items[:5]:
                            title = item.get_text(strip=True)
                            updates[category]["updates"].append({
                                "title": title,
                                "snippet": "",
                                "source": source_url
                            })
                    else:
                        # Generic extraction of headlines and links
                        headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], limit=5)
                        for headline in headlines:
                            text = headline.get_text(strip=True)
                            if len(text) > 20:  # Filter out short titles
                                updates[category]["updates"].append({
                                    "title": text,
                                    "snippet": "",
                                    "source": source_url
                                })
                
                except Exception as e:
                    st.error(f"Error scraping {source_url}: {str(e)}")
                    continue
    
    return updates

def fetch_document_compliance(document_text: str) -> Dict:
    """
    Generate compliance information relevant to the document type
    """
    document_categories = classify_document_type(document_text)
    
    # Take top 3 most likely categories for compliance
    top_categories = document_categories[:3]
    
    # Compliance requirements for different document types
    compliance_data = {}
    
    # Add compliance information for relevant categories
    for category, confidence in top_categories:
        if confidence < 0.5:  # Skip if confidence is very low
            continue
            
        compliance_data[category] = {
            "confidence": round(confidence, 2),
            "requirements": [],
            "relevant_regulations": [],
            "updates": []
        }
        
        # Add specific compliance requirements based on document type
        if category == "GDPR":
            compliance_data[category]["requirements"] = [
                "🛡️ Lawful basis for data processing documented",
                "📝 Clear privacy notice provided to data subjects",
                "🔒 Data minimization practices implemented",
                "⏱️ Right to erasure procedure established",
                "📤 Data portability mechanism available",
                "🕵️ Data Protection Impact Assessments conducted",
                "📞 Designated Data Protection Officer (if required)",
                "⚠️ 72-hour breach notification process in place"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "EU General Data Protection Regulation (GDPR)",
                "ePrivacy Directive",
                "National Data Protection Laws",
                "Cross-Border Data Transfer Regulations"
            ]
            
        elif category == "HIPAA":
            compliance_data[category]["requirements"] = [
                "🏥 Patient authorization for PHI disclosure",
                "📁 Minimum Necessary Standard implemented",
                "🔐 Physical and technical safeguards for ePHI",
                "📝 Notice of Privacy Practices displayed",
                "👥 Workforce security training conducted",
                "📅 6-year documentation retention policy",
                "🚨 Breach notification protocol established",
                "📊 Business Associate Agreements in place"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "Health Insurance Portability and Accountability Act (HIPAA)",
                "HITECH Act",
                "Omnibus Rule",
                "State Medical Privacy Laws"
            ]
            
        elif category == "Contracts":
            compliance_data[category]["requirements"] = [
                "📋 All parties properly identified and defined",
                "🔍 Scope of work/services clearly outlined",
                "💰 Payment terms and conditions specified",
                "⏱️ Performance timelines established",
                "🛑 Termination clauses included",
                "🔒 Confidentiality provisions included",
                "⚠️ Liability limitations specified",
                "⚖️ Governing law and dispute resolution"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "Uniform Commercial Code (UCC)",
                "State Contract Laws",
                "Electronic Signatures in Global and National Commerce Act",
                "Foreign Corrupt Practices Act (if international)"
            ]
            
        elif category == "Intellectual Property":
            compliance_data[category]["requirements"] = [
                "🔍 Clear definition of IP assets in question",
                "🔐 Ownership rights explicitly stated",
                "📝 License terms and restrictions detailed",
                "🌎 Territorial limitations specified",
                "⏱️ Duration of rights clearly stated",
                "💰 Royalty or compensation structure",
                "⚠️ Infringement remedies outlined",
                "🔄 Rights to derivatives and improvements"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "Copyright Act",
                "Patent Act",
                "Lanham Act (Trademarks)",
                "Defend Trade Secrets Act",
                "Digital Millennium Copyright Act"
            ]
            
        elif category == "Employment Law":
            compliance_data[category]["requirements"] = [
                "📝 Clear employment terms and conditions",
                "⏰ Working hours and overtime policies",
                "💰 Compensation and benefits structure",
                "🏥 Leave policies (sick, family, vacation)",
                "🔒 Non-disclosure and non-compete provisions",
                "⚠️ Termination procedures and severance",
                "🚫 Anti-discrimination policies",
                "👥 Employee classification (W2 vs 1099)"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "Fair Labor Standards Act (FLSA)",
                "Family and Medical Leave Act (FMLA)",
                "Title VII of Civil Rights Act",
                "Americans with Disabilities Act (ADA)",
                "Age Discrimination in Employment Act"
            ]
            
        elif category == "Real Estate":
            compliance_data[category]["requirements"] = [
                "📍 Property clearly identified and described",
                "💰 Purchase price and payment terms",
                "🔍 Property inspection contingencies",
                "🏠 Disclosures of known defects",
                "📝 Title examination and insurance",
                "💼 Closing costs allocation",
                "📅 Timeline for closing",
                "⚠️ Default and remedy provisions"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "State Property Laws",
                "Real Estate Settlement Procedures Act (RESPA)",
                "Truth in Lending Act (for financing)",
                "Fair Housing Act",
                "Local Zoning Ordinances"
            ]
            
        elif category == "Tax Law":
            compliance_data[category]["requirements"] = [
                "💵 Tax treatment of transactions specified",
                "📊 Record-keeping requirements",
                "🔄 Tax withholding and reporting obligations",
                "🌐 Cross-border tax considerations",
                "🏢 Entity classification for tax purposes",
                "💰 Tax indemnification provisions",
                "📝 Documentation for deductions/credits",
                "⚠️ Tax representation and warranties"
            ]
            compliance_data[category]["relevant_regulations"] = [
                "Internal Revenue Code",
                "State Tax Laws",
                "Foreign Account Tax Compliance Act (FATCA)",
                "Base Erosion and Profit Shifting (BEPS)",
                "Local Tax Regulations"
            ]
            
        # Fetch live updates for this category
        if category in LEGAL_UPDATE_SOURCES:
            for source_url in LEGAL_UPDATE_SOURCES[category][:1]:  # Just use first source for compliance updates
                try:
                    response = fetch_with_retry(source_url)
                    if not response:
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract recent updates for display
                    headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], limit=3)
                    for headline in headlines:
                        text = headline.get_text(strip=True)
                        if len(text) > 20:  # Filter out short titles
                            compliance_data[category]["updates"].append({
                                "title": text,
                                "source": source_url
                            })
                
                except Exception as e:
                    continue
    
    return compliance_data