#Importing Libraries
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#Downloading NLTK stopwords (if not already downloaded)
nltk.download('stopwords', quiet=True)
#Setting Paths    
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Models', 'logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'Models', 'tfidf_vectorizer.pkl')
# Initializing NLP assets (ONCE at the top level for maximum speed if used for a dataframe instead of a single text input)
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
def clean_text(text):
    """Cleans, tokenizes, and stems raw input text strings."""
    # Removing special characters and numbers, converting to lowercase
    text = text.lower() 
    text = re.sub(r'[0-9\W_]+', ' ', text)
    #Tokenizing and stemming the text
    tokens = text.split()
    tokens = [STEMMER.stem(token) for token in tokens if token not in STOP_WORDS]
    text = ' '.join(tokens)
    return text
def main():
    # Importing the trained model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    # Predicting the text
    text = input("Enter the text to predict: ")
    text = clean_text(text)
    feature_vector= vectorizer.transform([text])
    predictions = model.predict(feature_vector)
    for prediction in predictions:
        if prediction == 1:
            print("Spam")
        else:
            print("Ham")
if __name__ == "__main__":
    main()