import unittest
import sys
import os
import joblib

# Add project root and src to import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from preprocess import preprocess_text
from feature_extraction import get_tfidf_vectorizer
from predictor import predict_category
from app import create_app

class TestNewsClassifier(unittest.TestCase):

    def setUp(self):
        # Create a Flask test client for testing endpoints
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.testing = True

    # 1. test_preprocess_lowercase: Verify output is all lowercase
    def test_preprocess_lowercase(self):
        raw_text = "The QUICK Brown Fox"
        processed = preprocess_text(raw_text)
        self.assertEqual(processed, processed.lower())

    # 2. test_preprocess_url_removal: Verify URLs are stripped
    def test_preprocess_url_removal(self):
        raw_text = "Visit http://example.com or www.google.com for more information."
        processed = preprocess_text(raw_text)
        self.assertNotIn("http", processed)
        self.assertNotIn("www", processed)
        self.assertNotIn("examplecom", processed)

    # 3. test_preprocess_html_removal: Verify HTML tags are stripped
    def test_preprocess_html_removal(self):
        raw_text = "<p>This is a <b>bold</b> paragraph.</p>"
        processed = preprocess_text(raw_text)
        self.assertNotIn("<p>", processed)
        self.assertNotIn("<b>", processed)
        self.assertNotIn("</p>", processed)

    # 4. test_preprocess_stopword_removal: Verify common stopwords absent from output
    def test_preprocess_stopword_removal(self):
        raw_text = "This is a very simple sentence detailing the truth."
        processed = preprocess_text(raw_text).split()
        stopwords_list = {"this", "is", "a", "the"}
        for stopword in stopwords_list:
            self.assertNotIn(stopword, processed)

    # 5. test_preprocess_stemming: Verify "announced" maps to "announc"
    def test_preprocess_stemming(self):
        raw_text = "announced"
        processed = preprocess_text(raw_text)
        # NLTK PorterStemmer reduces 'announced' to 'announc'
        self.assertEqual(processed, "announc")

    # 6. test_vectoriser_no_leakage: Verify vectoriser vocabulary derived from training set only
    def test_vectoriser_no_leakage(self):
        # We initialize a new vectorizer
        vec = get_tfidf_vectorizer()
        train_corpus = ["politics senate house", "sports football goal"]
        test_corpus = ["technology software artificial"]
        
        # Fit on training corpus only
        vec.fit(train_corpus)
        vocab = vec.vocabulary_
        
        # Verify training terms are present
        self.assertIn("politics", vocab)
        self.assertIn("football", vocab)
        
        # Verify test-only terms are absent from vocabulary
        self.assertNotIn("technology", vocab)
        self.assertNotIn("software", vocab)
        
        # Transform test corpus (should work without changing vocabulary)
        test_vec = vec.transform(test_corpus)
        self.assertEqual(test_vec.shape[1], len(vocab))
        self.assertNotIn("technology", vec.vocabulary_)

    # 7. test_classifier_output_valid_label: Verify prediction is one of the five valid category labels
    def test_classifier_output_valid_label(self):
        sample = "The president signed the budget bill passed by the senators at the assembly."
        res = predict_category(sample)
        self.assertIn(res["category"], config.CATEGORIES)

    # 8. test_confidence_range: Verify confidence score is between 0.0 and 1.0
    def test_confidence_range(self):
        sample = "The national team won the football match at the stadium scoring three goals."
        res = predict_category(sample)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    # 9. test_empty_input_handling: Verify Flask returns 400 on empty input
    def test_empty_input_handling(self):
        # Post empty text to the /classify route
        response = self.client.post('/classify', data={'article_text': ''})
        self.assertEqual(response.status_code, 400)
        
        # Post whitespace text
        response_ws = self.client.post('/classify', data={'article_text': '   \n  '})
        self.assertEqual(response_ws.status_code, 400)

    # 10. test_health_endpoint: Verify /health returns 200 with correct JSON
    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "SVM")
        self.assertIn("vectoriser", data)

if __name__ == '__main__':
    unittest.main()
