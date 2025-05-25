import streamlit as st
from PyPDF2 import PdfReader

st.title("MCQ Generator")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded successfully✅")
else:
    st.info("Please upload a PDF file.")

