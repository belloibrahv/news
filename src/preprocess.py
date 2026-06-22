import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK resources are loaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    """
    Applies a 7-step preprocessing pipeline to raw news article text:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation and non-alphabetic characters
    5. Tokenize into words
    6. Remove English stopwords
    7. Stem tokens to their root forms using Porter Stemmer
    
    Args:
        text (str): Raw input news article text.
        
    Returns:
        str: Cleaned, space-separated string of stemmed tokens.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 3: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Step 4: Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Step 5: Tokenize
    tokens = word_tokenize(text)
    
    # Step 6: Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Step 7: Stem tokens
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Join tokens into a single space-separated string
    return ' '.join(stemmed_tokens)

def get_preprocessing_steps(text: str) -> dict:
    """
    Utility function for debugging / developer inspection of preprocessing stages (US-06).
    Returns a dictionary mapping step number/description to intermediate string values.
    """
    steps = {}
    
    # Step 1: Lowercase
    step1 = text.lower()
    steps["1_lowercase"] = step1
    
    # Step 2: URL Removal
    step2 = re.sub(r'http\S+|www\S+', '', step1)
    steps["2_url_removal"] = step2
    
    # Step 3: HTML Tag Removal
    step3 = re.sub(r'<.*?>', '', step2)
    steps["3_html_removal"] = step3
    
    # Step 4: Non-alphabetic Removal
    step4 = re.sub(r'[^a-z\s]', '', step3)
    steps["4_punctuation_removal"] = step4
    
    # Step 5: Tokenization
    step5 = word_tokenize(step4)
    steps["5_tokenization"] = step5
    
    # Step 6: Stopword Removal
    step6 = [token for token in step5 if token not in stop_words]
    steps["6_stopword_removal"] = step6
    
    # Step 7: Stemming
    step7 = [stemmer.stem(token) for token in step6]
    steps["7_stemming"] = step7
    
    # Final Join
    steps["final_output"] = ' '.join(step7)
    
    return steps
