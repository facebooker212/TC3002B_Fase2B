"""
Mayra Fernanda Camacho Rodriguez A01378998
Victor Martinez Roman A01746361
Fase 2 - Parte B
Para las distintas secciones y módulos, se especificará con un comentario de línea
el inicio y el final de cada sección. Las secciones especificadas en la fase 1 son:
1 - Preprocesamiento y limpieza
2 - Tokenizar y vectorizar el texto
3 - Comparación y determinación de plagio
4 - Visualización de resultados y comprobación
Se explicará dentro de cada función con un comentario de bloque su funcionamiento individual.
"""
import os
import re
import numpy as np
import datetime
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Guardamos los resultados y limites globales para levenshtein y coseno
levenshtein_scores = []
cosine_scores = []
labels = []

lev_threshold = 0.7
cos_threshold = 0.6

# 1- Preprocesamiento y limpieza
def preprocess_text(text):
    """
    Autor: Victor Martinez
    Dividimos el texto en oraciones, quitamos palabras de 2 letras o menos
    y borramos caracteres especiales, solo dejamos letras y números.
    """
    text = text.lower()
    
    sentences = text.split('.')
    
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sentences = [s for s in sentences if len(s) >= 2]
    
    sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', s) for s in sentences]
    
    return sentences

def stem_text(text):
    """
    Autor: Mayra Fernanda Camacho
    Aplica stemming usando la libreria nltk y devuelve el texto procesado
    """
    stemmer = PorterStemmer()

    words = text.split()

    stemmed_words = [stemmer.stem(word) for word in words]

    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text
# Termina preprocesamiento y limpieza

# 2 - Tokenizacion del texto
def generate_ngrams(text, n=3):
    """
    Autor: Victor Martinez
    Genera n-gramas, por defecto trigramas y devuelve el texto procesado
    solo se usa tokenizacion en similitud de coseno.
    """
    words = text.split()

    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    return ngrams
# Termina tokenizacion del texto

# 3 - Comparación y determinación de plagio 
def calculate_levenshtein_similarity(text1, text2):
    """
    Autor: Mayra Fernanda Camacho
    Similitud de Levenshtein solo con texto pre-procesado (sin stemming ni n-gramas)
    dado que queremos el texto sin modificar solo para Levenshtein.
    """
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    max_len = max(len(preprocessed_text1), len(preprocessed_text2))

    similarity_sum = 0
    suspected_portions = []
    for i in range(min(len(preprocessed_text1), len(preprocessed_text2))):
        sentence1 = preprocessed_text1[i]
        sentence2 = preprocessed_text2[i]
        similarity = 1 - (levenshtein_distance(sentence1, sentence2) / max(len(sentence1), len(sentence2)))
        similarity_sum += similarity
        if similarity >= lev_threshold:
            suspected_portions.append((sentence1, sentence2))

    similarity_score = similarity_sum / max_len
    return similarity_score, suspected_portions

def calculate_cosine_similarity(text1, text2):
    """
    Autor: Victor Martinez
    Calcula similitud de coseno y vectoriza (solo se vectoriza en similitud de coseno)
    aquí pasa por preprocesamiento, stemming y n-gramas.
    """
    stemmed_text1 = stem_text(text1)
    stemmed_text2 = stem_text(text2)

    preprocessed_text1 = preprocess_text(stemmed_text1)
    preprocessed_text2 = preprocess_text(stemmed_text2)

    ngrams1 = generate_ngrams(' '.join(preprocessed_text1))
    ngrams2 = generate_ngrams(' '.join(preprocessed_text2))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(ngrams1 + ngrams2)
    similarity_matrix = cosine_similarity(X[:len(ngrams1)], X[len(ngrams1):])

    similarity_score = similarity_matrix[0][0]
    suspected_portions = []
    if similarity_score >= cos_threshold:
        suspected_portions = list(zip(preprocessed_text1, preprocessed_text2))

    return similarity_score, suspected_portions

def detect_plagiarism(source_dir, target_dir):
    """
    Autor: En conjunto
    Lee todos los archivos en los directorios especificados, aplica levenshtein 
    y similitud de coseno, devuelve los resultados de estos dos métodos de comparación.
    """
    similarity_results = {}
    
    for source_filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, source_filename)
        with open(source_file_path, 'r') as source_file:
            source_text = source_file.read()
        
        for target_filename in os.listdir(target_dir):
            target_file_path = os.path.join(target_dir, target_filename)
            with open(target_file_path, 'r', encoding='latin-1') as target_file:
                target_text = target_file.read()
            
            levenshtein_score, lev_suspected_portions = calculate_levenshtein_similarity(source_text, target_text)
            cosine_score, cos_suspected_portions = calculate_cosine_similarity(source_text, target_text)
           
            suspected_portions = lev_suspected_portions + cos_suspected_portions

            similarity_results[(source_filename, target_filename)] = {
                'levenshtein_similarity': levenshtein_score,
                'cosine_similarity': cosine_score,
                'suspected_portions': suspected_portions
            }

    return similarity_results
# Termina comparación y determinación de plagio

# 4 - Visualización de resultados y comprobación
def save_plagiarism_report(plagiarism_scores):
    """
    Autor: En conjunto
    Guarda los resultados en un reporte en formato .txt, guarda porcentajes y fragmentos plagiados
    """
    output_filename = f"plagiarism_report_{datetime.datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(output_filename, 'w') as output_file:
        for (source_filename, target_filename), similarity_scores in plagiarism_scores.items():
            levenshtein_percentage = similarity_scores['levenshtein_similarity'] * 100
            cosine_percentage = similarity_scores['cosine_similarity'] * 100
            levenshtein_scores.append(similarity_scores['levenshtein_similarity'])
            cosine_scores.append(similarity_scores['cosine_similarity'])
            if levenshtein_percentage >= lev_threshold * 100 or cosine_percentage >= cos_threshold * 100:
                output_file.write(f"{source_filename} vs {target_filename}:\n")
                output_file.write(f"Levenshtein Similarity: {levenshtein_percentage:.2f}%\n")
                output_file.write(f"Cosine Similarity: {cosine_percentage:.2f}%\n")
                for suspected_portion in similarity_scores['suspected_portions']:
                    output_file.write(f"Original: {suspected_portion[1]}\n")
                    output_file.write(f"Suspected Plagiarism: {suspected_portion[0]}\n\n")

    print(f"Plagiarism report saved to {output_filename}")

def calculate_metrics(levenshtein_scores, cosine_scores, labels):
    """
    Autor: Mayra Fernanda Camacho
    Usando los resultados de ambas comparaciones y la clasificación 
    absoluta (definida abajo) de plagio u original, calcula valores para: TPR, FPR y AUC.
    """
    lev_fpr, lev_tpr, _ = roc_curve(labels, levenshtein_scores)
    lev_auc = auc(lev_fpr, lev_tpr)
    
    cos_fpr, cos_tpr, _ = roc_curve(labels, cosine_scores, pos_label=1)
    cos_auc = auc(cos_fpr, cos_tpr)
    
    return {
        'levenshtein': {
            'fpr': lev_fpr,
            'tpr': lev_tpr,
            'auc': lev_auc
        },
        'cosine': {
            'fpr': cos_fpr,
            'tpr': cos_tpr,
            'auc': cos_auc
        }
    }

def classify_plagiarism(levenshtein_scores, cosine_scores):
    """
    Autor: Victor Martinez
    Toma los resultados de ambos métodos, y 2 métricas para determinar por método 
    si un documento es plagiado o no, combina ambos métodos y determina un resultado 
    en conjunto (para distintos posibles escenarios de plagio).
    Devuelve una lista con solo 1 y 0 (plagio u original).
    """
    classifications = []
    for lev_score, cos_score in zip(levenshtein_scores, cosine_scores):
        if lev_score >= lev_threshold or cos_score >= cos_threshold:
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications

def plot_roc_curve(metrics):
    """
    Autor: En conjunto
    Gráfica AUC usando las métricas calculadas anteriormente para Levenshtein y similitud de coseno.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['levenshtein']['fpr'], metrics['levenshtein']['tpr'], label=f'Levenshtein (AUC = {metrics["levenshtein"]["auc"]:.2f})')
    plt.plot(metrics['cosine']['fpr'], metrics['cosine']['tpr'], label=f'Cosine (AUC = {metrics["cosine"]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
# Termina visualización de resultados y comprobación 

if __name__ == '__main__':
    plagiarism_scores = detect_plagiarism('./suspicious/', './original/')
    save_plagiarism_report(plagiarism_scores)
    labels = classify_plagiarism(levenshtein_scores, cosine_scores)
    metrics = calculate_metrics(levenshtein_scores, cosine_scores, labels)
    plot_roc_curve(metrics)
