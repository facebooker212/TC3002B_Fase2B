import os
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio
from helper_spacy import detect_plagiarism_technique

# Pre procesamiento
def preprocess_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin-1') as file:
            document = file.read()
            document = re.sub(r'[^a-zA-Z0-9\s]', '', document)
            tokens = word_tokenize(document.lower())
            tokens = [word for word in tokens if len(word) >= 3]
            documents.append(' '.join(tokens))
    return documents

# Levenshtein re-implementado
def calculate_levenshtein_similarity(original_doc, suspicious_doc):
    return levenshtein_ratio(original_doc.lower(), suspicious_doc.lower())

# Function to detect plagiarism type
def detect_plagiarism_type(original_doc, suspicious_doc, cosine_sim, levenshtein_score):
    plagiarism_technique = detect_plagiarism_technique(original_doc, suspicious_doc)
    if plagiarism_technique != "No":
        return plagiarism_technique
    if cosine_sim < levenshtein_score:
        if (levenshtein_score - cosine_sim) > 0.2:
            return "Parafraseo"
        else:
            return "Insertar o reemplazar frases"
    elif cosine_sim > levenshtein_score:
        return "Desordenar las frases"
    else:
        return "Unknown"

def detect_plagiarism(similarity_scores, levenshtein_scores, original_docs, suspicious_docs, threshold=0.7, levenshtein_threshold=0.7):
    plagiarism_results = []
    for i, suspicious_doc in enumerate(suspicious_docs):
        is_copy = "No"
        original_file = "----------"
        plagiarism_percentage = "----------"
        plagiarism_type = "----------"
        for j, original_doc in enumerate(original_docs):
            cosine_sim = similarity_scores[i][j]
            levenshtein_score = levenshtein_scores[i][j]
            if cosine_sim > threshold or levenshtein_score > levenshtein_threshold:
            #if True:
                is_copy = "Si"
                original_file = os.listdir(original_folder)[j]
                plagiarism_percentage = round(cosine_sim * 100, 2)
                plagiarism_type = detect_plagiarism_type(original_doc, suspicious_doc, cosine_sim, levenshtein_score)
                match_details = {
                    "suspicious_file": os.listdir(suspicious_folder)[i],
                    "is_copy": is_copy,
                    "original_file": original_file,
                    "plagiarism_percentage": plagiarism_percentage,
                    "plagiarism_type": plagiarism_type
                }
                plagiarism_results.append(match_details)

        if is_copy == "No":
            match_details = {
                "suspicious_file": os.listdir(suspicious_folder)[i],
                "is_copy": is_copy,
                "original_file": original_file,
                "plagiarism_percentage": plagiarism_percentage,
                "plagiarism_type": plagiarism_type
            }
            plagiarism_results.append(match_details)

    return plagiarism_results

def main(original_folder, suspicious_folder):
    original_docs = preprocess_documents(original_folder)
    suspicious_docs = preprocess_documents(suspicious_folder)

    # Tokenizacion y vectorizacion
    vectorizer = TfidfVectorizer()
    original_features = vectorizer.fit_transform(original_docs)
    suspicious_features = vectorizer.transform(suspicious_docs)

    # Similitud de coseno pero entre todo el documento
    similarity_scores = cosine_similarity(suspicious_features, original_features)

    # Levenshtein re-implementado
    levenshtein_scores = []
    for suspicious_doc in suspicious_docs:
        levenshtein_scores.append([calculate_levenshtein_similarity(original_doc, suspicious_doc) for original_doc in original_docs])
    
    # Calcula plagio
    plagiarism_results = detect_plagiarism(similarity_scores, levenshtein_scores, original_docs, suspicious_docs)

    # Resultados
    with open("plagiarism_report.txt", "w") as f:
        f.write("Documento sospechoso\tCopia\tDocumento plagiado\t% Plagio\tTipo de plagio\n")
        for result in plagiarism_results:
            f.write(f"{result['suspicious_file']}\t\t{result['is_copy']}\t\t{result['original_file']}\t\t{result['plagiarism_percentage']}\t\t\t{result['plagiarism_type']}\n")
    print("Documento sospechoso\tCopia\tDocumento plagiado\t% Plagio\tTipo de plagio")
    for result in plagiarism_results:
        print(f"{result['suspicious_file']}\t\t{result['is_copy']}\t{result['original_file']}\t\t{result['plagiarism_percentage']}\t\t{result['plagiarism_type']}")

if __name__ == "__main__":
    #try:
    #    nltk.find('punkt')
    #except:
    #    nltk.download('punkt')
    original_folder = "./original/"
    suspicious_folder = "./suspicious_2/"
    main(original_folder, suspicious_folder)

