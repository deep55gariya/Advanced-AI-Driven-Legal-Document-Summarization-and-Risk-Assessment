import requests
from bs4 import BeautifulSoup
import streamlit as st

def fetch_compliance_guidelines():
    """Enhanced compliance checklist with structured requirements"""
    checklists = {
        "GDPR": {
            "checklist": [
                "🛡️ Lawful basis for data processing documented",
                "📝 Clear privacy notice provided to data subjects",
                "🔒 Data minimization practices implemented",
                "⏱️ Right to erasure procedure established",
                "📤 Data portability mechanism available",
                "🕵️ Data Protection Impact Assessments conducted",
                "📞 Designated Data Protection Officer (if required)",
                "⚠️ 72-hour breach notification process in place"
            ],
            "reference": "https://gdpr-info.eu/"
        },
        "HIPAA": {
            "checklist": [
                "🏥 Patient authorization for PHI disclosure",
                "📁 Minimum Necessary Standard implemented",
                "🔐 Physical and technical safeguards for ePHI",
                "📝 Notice of Privacy Practices displayed",
                "👥 Workforce security training conducted",
                "📅 6-year documentation retention policy",
                "🚨 Breach notification protocol established",
                "📊 Business Associate Agreements in place"
            ],
            "reference": "https://www.hhs.gov/hipaa/"
        }
    }

    try:
        # Add live updates from official sources
        for name in checklists.keys():
            try:
                response = requests.get(
                    checklists[name]["reference"],
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=5
                )
                soup = BeautifulSoup(response.text, 'html.parser')

                updates = []
                if name == "GDPR":
                    articles = soup.find_all('article', limit=3)
                    updates = [a.get_text(strip=True) for a in articles if a.get_text(strip=True)]
                elif name == "HIPAA":
                    content = soup.find('div', {'class': 'content'})
                    updates = [p.get_text(strip=True) for p in content.find_all('p', limit=3)] if content else []

                checklists[name]["latest_updates"] = updates

            except Exception as e:
                checklists[name]["latest_updates"] = [f"⚠️ Failed to retrieve live updates: {str(e)}"]

        return checklists

    except Exception as e:
        return {"error": f"Compliance system unavailable: {str(e)}"}