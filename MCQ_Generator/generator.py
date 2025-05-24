import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import random

# Load models
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

def get_top_sentences(text, top_n=5):
    sentences = sent_tokenize(text)
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = tfidf.sum(axis=1).A1
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [sentences[i] for i in top_indices]

def extract_named_entities(sentence):
    doc = nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]

def generate_distractors(correct_entity, entity_type, all_entities):
    same_type = [ent for ent, typ in all_entities if typ == entity_type and ent != correct_entity]
    distractors = random.sample(same_type, k=min(3, len(same_type))) if same_type else []
    return distractors

def create_mcq(sentence, entity, distractors):
    question = sentence.replace(entity, "_____")
    options = distractors + [entity]
    random.shuffle(options)
    return {
        "question": question,
        "options": options,
        "answer": entity
    }

def generate_mcqs_from_text(text):
    mcqs = []
    top_sentences = get_top_sentences(text, top_n=10)

    # Collect all entities from top sentences for distractor pool
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
