import unittest
import os
from nltk.stem import PorterStemmer
from main import (
    detect_plagiarism,
    calculate_levenshtein_similarity,
    calculate_cosine_similarity,
    calculate_metrics,
    classify_plagiarism,
    generate_ngrams,
    stem_text,
    preprocess_text,
)

class TestPlagiarismDetection(unittest.TestCase):
    def setUp(self):
        self.original_dir = "tests/original"
        self.suspicious_dir = "tests/suspicious"
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.suspicious_dir, exist_ok=True)

    def tearDown(self):
        for dir_path in [self.original_dir, self.suspicious_dir]:
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                os.remove(file_path)
            os.rmdir(dir_path)

    def test_identical_texts(self):
        original_text = "This is a long sample text for testing the plagiarism detection system. It contains multiple sentences and should be long enough to accurately evaluate the similarity measures. The text discusses a general topic and does not contain any specific information that could be plagiarized. This is another sentence to increase the length of the text."
        self.create_file(self.original_dir, "original.txt", original_text)
        self.create_file(self.suspicious_dir, "suspicious.txt", original_text)

        scores = detect_plagiarism(self.suspicious_dir, self.original_dir)
        self.assertEqual(len(scores), 1)
        for score in scores.values():
            self.assertGreaterEqual(score["levenshtein_similarity"], 0.9)
            self.assertGreaterEqual(score["cosine_similarity"], 0.9)

    def test_completely_different_texts(self):
        original_text = "Este es un texto que habla acerca de un tema completamente distinto al otro. Este texto habla sobre Inteligencia Artificial y solo eso. Este es el final."
        suspicious_text = "Ahora podemos conversar sobre otro topico. Este fragmento propone discutir Machine Learning. No es el todo relacionado a la Inteligencia Artificial. Aqui termina."
        self.create_file(self.original_dir, "original.txt", original_text)
        self.create_file(self.suspicious_dir, "suspicious.txt", suspicious_text)

        scores = detect_plagiarism(self.suspicious_dir, self.original_dir)
        self.assertEqual(len(scores), 1)
        for score in scores.values():
            self.assertLess(score["levenshtein_similarity"], 0.5)
            self.assertLess(score["cosine_similarity"], 0.5)

    def test_partial_plagiarism(self):
        original_text = "This is a long sample text for testing the plagiarism detection system. It contains multiple sentences and should be long enough to accurately evaluate the similarity measures. The text discusses a general topic and does not contain any specific information that could be plagiarized. This is another sentence to increase the length of the text."
        plagiarized_text = "This is a long sample text for testing the plagiarism detection system. It contains multiple sentences and should be long enough to accurately evaluate the similarity measures. The text discusses a different topic and contains some plagiarized information. This is another sentence to increase the length of the text."
        self.create_file(self.original_dir, "original.txt", original_text)
        self.create_file(self.suspicious_dir, "suspicious.txt", plagiarized_text)

        scores = detect_plagiarism(self.suspicious_dir, self.original_dir)
        self.assertEqual(len(scores), 1)
        for score in scores.values():
            self.assertLess(score["levenshtein_similarity"], 1.0)
            self.assertLess(score["cosine_similarity"], 1.0)
            self.assertGreater(score["levenshtein_similarity"], 0.0)
            self.assertGreater(score["cosine_similarity"], 0.0)

    def test_levenshtein_similarity(self):
        text1 = "This is a sample text."
        text2 = "This is a simple text."
        similarity, _ = calculate_levenshtein_similarity(text1, text2)
        self.assertGreaterEqual(similarity, 0.7)

    def test_cosine_similarity(self):
        text1 = "This is a sample text."
        text2 = "This is another sample text."
        similarity, _ = calculate_cosine_similarity(text1, text2)
        self.assertGreaterEqual(similarity, 0.6)

    def test_calculate_metrics(self):
        levenshtein_scores = [0.8, 0.6, 0.9]
        cosine_scores = [0.7, 0.5, 0.8]
        labels = [1, 0, 1]
        metrics = calculate_metrics(levenshtein_scores, cosine_scores, labels)
        self.assertIsInstance(metrics, dict)
        self.assertIn("levenshtein", metrics)
        self.assertIn("cosine", metrics)

    def test_classify_plagiarism(self):
        levenshtein_scores = [0.8, 0.6, 0.9]
        cosine_scores = [0.7, 0.5, 0.8]
        classifications = classify_plagiarism(levenshtein_scores, cosine_scores)
        self.assertEqual(len(classifications), 3)
        self.assertIn(0, classifications)
        self.assertIn(1, classifications)

    def test_generate_ngrams(self):
        text = "This is a sample text."
        ngrams = generate_ngrams(text)
        expected_ngrams = ["This is a", "is a sample", "a sample text."]
        self.assertEqual(ngrams, expected_ngrams)

    def test_stem_text(self):
        text = "This is a sample text."
        stemmed_text = stem_text(text)
        expected_stemmed_text = "thi is a sampl text."
        self.assertEqual(stemmed_text, expected_stemmed_text)

    def test_preprocess_text(self):
        text = "This is a sample text! with, some? punctuation."
        preprocessed_text = preprocess_text(text)
        expected_preprocessed_text = ["this is a sample text with some punctuation"]
        self.assertEqual(preprocessed_text, expected_preprocessed_text)

    def create_file(self, dir_path, file_name, content):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, "w") as file:
            file.write(content)

if __name__ == "__main__":
    unittest.main()
