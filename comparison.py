import difflib
import streamlit as st

def compare_documents(doc1: str, doc2: str) -> str:
    """Document comparison with error handling"""
    try:
        d = difflib.Differ()
        diff = list(d.compare(doc1.splitlines(), doc2.splitlines()))

        result = []
        for line in diff:
            if line.startswith('- '):
                result.append(f'<span style="color:red">{line}</span>')
            elif line.startswith('+ '):
                result.append(f'<span style="color:green">{line}</span>')
            else:
                result.append(line)

        return '<br>'.join(result)
    except Exception as e:
        return f"Comparison failed: {str(e)}"
