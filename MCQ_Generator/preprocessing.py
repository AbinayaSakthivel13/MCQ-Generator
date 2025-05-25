import spacy
from PyPDF2 import PdfReader
from collections import Counter
from frontend import extract_text_from_pdf

def extract_text_by_page(pdf_file):
    """
    Extracts text from each page of a PDF as a list of line lists.
    """
    reader = PdfReader(pdf_file)
    all_pages_lines = []

    for page in reader.pages:
        text = page.extract_text()
        lines = text.splitlines() if text else []
        all_pages_lines.append(lines)

    return all_pages_lines

def remove_repeated_lines(pages_text, min_line_length=20):
    """
    Removes repeated headers/footers and filters irrelevant short lines.
    """
    header_candidates = []
    footer_candidates = []

    for lines in pages_text:
        if len(lines) >= 2:
            header_candidates.append(lines[0].strip())
            footer_candidates.append(lines[-1].strip())

    # Count occurrences
    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)

    total_pages = len(pages_text)
    common_headers = {line for line, count in header_counts.items() if count > total_pages // 2}
    common_footers = {line for line, count in footer_counts.items() if count > total_pages // 2}

    cleaned_pages = []
    for lines in pages_text:
        cleaned = [
            line.strip() for line in lines
            if line.strip() not in common_headers
            and line.strip() not in common_footers
            and len(line.strip()) > min_line_length
        ]
        cleaned_pages.append(" ".join(cleaned))

    return "\n\n".join(cleaned_pages)

# === Usage ===
pdf_file=extract_text_from_pdf()

pages_text = extract_text_by_page(pdf_file)
cleaned_text = remove_repeated_lines(pages_text)

# Optional: save or print result
with open("cleaned_output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("✅ PDF text cleaned and saved to cleaned_output.txt")

#Segementation and Tokenization
# This code segments the cleaned text into sentences and tokenizes each sentence using spaCy.

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def segment_and_tokenize(text):
    """
    Segments text into sentences and tokenizes each sentence.
    Returns a list of sentence strings and a list of token lists.
    """
    doc = nlp(text)

    sentences = []
    tokenized_sentences = []

    for sent in doc.sents:
        sentences.append(sent.text.strip())
        tokenized_sentences.append([token.text for token in sent])

    return sentences, tokenized_sentences

# === Example usage ===
with open("cleaned_output.txt", "r", encoding="utf-8") as f:
    cleaned_text = f.read()

sentences, tokens = segment_and_tokenize(cleaned_text)

# Show first few results
print("\n=== Sample Sentences ===")
for i, s in enumerate(sentences[:5]):
    print(f"{i+1}: {s}")

print("\n=== Sample Tokens from First Sentence ===")
print(tokens[0])

# Identify Question-Worthy Sentences
# This code applies heuristic rules to identify sentences that are worth turning into questions.

def is_question_worthy(sent):
    """
    Apply heuristic rules to determine if a sentence is worth turning into a question.
    """
    text = sent.text.strip()
    if len(text.split()) < 5:
        return False  # too short

    has_named_entity = any(ent.label_ in ("PERSON", "ORG", "GPE", "DATE", "MONEY", "EVENT") for ent in sent.ents)
    has_number = any(tok.like_num for tok in sent)
    is_definition = any(phrase in text.lower() for phrase in [" is ", " are ", " refers to", " defined as", " means "])
    has_fact_verb = any(tok.lemma_ in ("invent", "discover", "create", "establish", "develop", "found") for tok in sent)

    return has_named_entity or has_number or is_definition or has_fact_verb

def extract_question_worthy_sentences(text):
    """
    Runs heuristics on all sentences in the text to return a filtered list.
    """
    doc = nlp(text)
    question_worthy = []

    for sent in doc.sents:
        if is_question_worthy(sent):
            question_worthy.append(sent.text.strip())

    return question_worthy

# === Example Usage ===
with open("cleaned_output.txt", "r", encoding="utf-8") as f:
    cleaned_text = f.read()

important_sents = extract_question_worthy_sentences(cleaned_text)

# Show top 10 results
print("\n=== Question-Worthy Sentences ===")
for i, s in enumerate(important_sents[:10]):
    print(f"{i+1}. {s}")
with open("ques_worthy_sents.txt", "w", encoding="utf-8") as f:
    f.write(important_sents)
print("\n✅ Question-worthy sentences saved to ques_worthy_sents.txt")