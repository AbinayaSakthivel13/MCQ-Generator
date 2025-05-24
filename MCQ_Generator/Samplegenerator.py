import streamlit as st
from PyPDF2 import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# --- Sentence Ranking with TF-IDF (using spaCy for splitting) ---
def get_top_sentences(text, top_n=5):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 30]
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [sentences[i] for i in top_indices]

# --- NER ---
def extract_named_entities(sentence):
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]

# --- Distractor Generator ---
def generate_distractors(correct_entity, entity_type, all_entities):
    same_type = [ent for ent, typ in all_entities if typ == entity_type and ent != correct_entity]
    distractors = random.sample(same_type, k=min(3, len(same_type))) if same_type else []
    return distractors

# --- MCQ Creator ---
def create_mcq(sentence, entity, distractors):
    question = sentence.replace(entity, "_____")
    options = distractors + [entity]
    random.shuffle(options)
    return {
        "question": question,
        "options": options,
        "answer": entity
    }

# --- Full MCQ Generator Pipeline ---
def generate_mcqs_from_text(text):
    mcqs = []
    top_sentences = get_top_sentences(text, top_n=10)
    all_entities = []
    for sent in top_sentences:
        all_entities.extend(extract_named_entities(sent))

    for sent in top_sentences:
        entities = extract_named_entities(sent)
        for ent_text, ent_type in entities:
            distractors = generate_distractors(ent_text, ent_type, all_entities)
            if distractors:
                mcq = create_mcq(sent, ent_text, distractors)
                mcqs.append(mcq)
    return mcqs

# --- Streamlit UI ---
st.title("📘 MCQ Generator from PDF (No NLTK)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    mcqs = generate_mcqs_from_text(text)

    # Store for later use
    st.session_state["mcqs"] = mcqs

    st.success(f"✅ Generated {len(mcqs)} MCQs from your PDF.")

    if st.checkbox("Show Sample Questions"):
        for q in mcqs[:5]:
            st.markdown(f"**Q:** {q['question']}")
            for opt in q["options"]:
                st.markdown(f"- {opt}")
            st.markdown(f"✅ **Answer:** {q['answer']}")
            st.markdown("---")
else:
    st.info("Please upload a PDF to begin.")
