import spacy

# Modelo en ingles
nlp = spacy.load("en_core_web_sm")

def detect_plagiarism_technique(original_doc, suspicious_doc):
    if isinstance(original_doc, str):
        original_doc = nlp(original_doc)
    if isinstance(suspicious_doc, str):
        suspicious_doc = nlp(suspicious_doc)

    original_verbs = [token.lemma_ for token in original_doc if token.pos_ == "VERB"]
    suspicious_verbs = [token.lemma_ for token in suspicious_doc if token.pos_ == "VERB"]
    
    original_voice = [token.dep_ for token in original_doc if token.pos_ == "VERB"]
    suspicious_voice = [token.dep_ for token in suspicious_doc if token.pos_ == "VERB"]

    common_verbs = set(original_verbs) & set(suspicious_verbs)
    common_voice = set(original_voice) & set(suspicious_voice)

    verb_change_percentage = len(common_verbs) / len(suspicious_verbs) * 100 if suspicious_verbs else 0
    voice_change_percentage = len(common_voice) / len(suspicious_voice) * 100 if suspicious_voice else 0

    if verb_change_percentage > 90:
        if voice_change_percentage >= verb_change_percentage:
            return "Cambio de voz"
        else:
            return "Cambio de tiempo"
    elif voice_change_percentage > 90:
        return "Cambio de voz"
    else:
        return "No"

