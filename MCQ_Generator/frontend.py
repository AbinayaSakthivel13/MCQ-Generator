import streamlit as st
from PyPDF2 import PdfReader

st.title("📄 PDF Processor")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
def process_pdf_file(uploaded_file):
    """
    Processes the uploaded PDF and returns cleaned text as a single string.
    """
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Optional: simple cleaning (e.g., remove short lines)
    lines = full_text.splitlines()
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 30]
    cleaned_text = " ".join(cleaned_lines)

    return cleaned_text  # ready for NLP, question generation, etc.

if uploaded_file:
    # Just process and store the result internally
    cleaned_text = process_pdf_file(uploaded_file)

    # You now have `cleaned_text` ready to use in memory
    # No need to display anything yet
    st.success("File processed successfully and text stored.")
    
else:
    st.info("Please upload a PDF file.")

