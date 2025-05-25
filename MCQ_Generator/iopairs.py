import spacy

nlp = spacy.load("en_core_web_sm")

def classify_question_type(sentence):
    """
    Enhanced hybrid classifier using SpaCy parsing + keyword heuristics
    to classify sentence as MCQ, TF, or AR.
    """
    doc = nlp(sentence)
    text = sentence.lower().strip()

    # Keyword-based AR triggers
    causal_keywords = ["because", "due to", "as a result", "therefore", "since", "so", "hence", "thus"]
    causal_markers = {"because", "since", "as", "although", "though", "due to", "so", "therefore", "thus"}

    # Definition-based MCQ triggers (extended)
    definition_phrases = [
        " is the ", " are the ", " refers to ", " is defined as ", " can be defined as ",
        " known as ", " means ", " is called ", " is known as "
    ]

    has_causal_clause = any(token.dep_ == "mark" and token.text.lower() in causal_markers for token in doc)
    has_causal_keyword = any(kw in text for kw in causal_keywords)
    has_definition_pattern = any(phrase in text for phrase in definition_phrases)

    word_count = len(text.split())

    # Decision tree
    if has_causal_clause or has_causal_keyword:
        return "AR"
    elif has_definition_pattern:
        return "MCQ"
    elif text.endswith(".") and word_count >= 7:
        return "TF"
    else:
        return "MCQ"

def generate_output(sentence, q_type):

    doc = nlp(sentence)
    questions = []

    ENTITY_QUESTION_TEMPLATES = {
        "PERSON": "Who is {ent}?",
        "ORG": "What is {ent}?",
        "PRODUCT": "What is {ent}?",
        "EVENT": "What is {ent}?",
        "WORK_OF_ART": "What is {ent}?",
        "GPE": "Where is {ent}?",
        "LOC": "Where is {ent}?",
        "DATE": "When did {sentence}?",
        "TIME": "When did {sentence}?",
        "MONEY": "How much is {ent}?",
        "QUANTITY": "How much is {ent}?",
        "PERCENT": "How much is {ent}?",
        "ORDINAL": "What is the order of {ent}?",
        "CARDINAL": "What is the number of {ent}?",
        "LAW": "What is the law regarding {ent}?",
        "LANGUAGE": "What is the language of {ent}?",
        "NORM": "What is the norm regarding {ent}?",
        "FAC": "What is the facility of {ent}?",
        "MISC": "What is the miscellaneous information about {ent}?",
    }

    for ent in doc.ents:
        ent_label = ent.label_

        if ent_label in ENTITY_QUESTION_TEMPLATES:
            template = ENTITY_QUESTION_TEMPLATES[ent_label]

            if "{sentence}" in template:
                blanked = sentence.replace(ent.text, "_____")
                question = template.format(ent=ent.text, sentence=blanked.strip())
            else:
                question = template.format(ent=ent.text)

            if question not in questions:
                questions.append(question)

    # Fallback if no entity question was generated
    if not questions:
        questions.append(generate_generic_question(sentence))

    # Handle question type output
    if q_type == "TF":
        return f"{sentence.strip()} (True/False)"
    
    elif q_type == "AR":
        parts = sentence.split(" because ")
        if len(parts) == 2:
            assertion = parts[0].strip().rstrip(".")
            reason = parts[1].strip().rstrip(".")
            return f"Assertion: {assertion}. Reason: {reason}."
        else:
            return f"Assertion: {sentence.strip()}. Reason: [Unknown]."

    # Default: return the list of generated questions
    return questions


def generate_generic_question(sentence):
    lowered = sentence.lower()
    if " is the " in lowered:
        parts = sentence.split(" is the ")
        if len(parts) == 2:
            return f"What is the {parts[1].strip().rstrip('.') }?"
    elif " are the " in lowered:
        parts = sentence.split(" are the ")
        if len(parts) == 2:
            return f"What are the {parts[1].strip().rstrip('.') }?"
    return f"What is meant by: \"{sentence.strip()}\"?"


def create_io_pairs(sentences):
    """
    From a list of sentences, return a list of (Input, Output, Type) rows.
    """
    io_pairs = []
    for sent in sentences:
        q_type = classify_question_type(sent)
        output = generate_output(sent, q_type)
        io_pairs.append((sent.strip(), output.strip(), q_type))
    return io_pairs

with open("ques_worthy_sents.txt", "r", encoding="utf-8") as f:
    question_worthy_sentences = f.read()

pairs = create_io_pairs(question_worthy_sentences)

# Print as table
print(f"{'Input (Context)':<60} | {'Output (Question)':<60} | Type")
print("-" * 150)
for inp, outp, typ in pairs[:5]:  # Only first N items
    print(f"{inp:<60} | {outp:<60} | {typ}")