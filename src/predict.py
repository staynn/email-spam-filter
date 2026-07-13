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
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'models', 'tfidf_vectorizer.pkl')
# Initializing NLP assets (ONCE at the top level for maximum speed if used for a dataframe instead of a single text input)
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
# Importing the trained model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
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
def predict_message(message):
    """Predicts whether a given message is spam or not."""
    cleaned_message = clean_text(message)
    feature_vector= vectorizer.transform([cleaned_message])
    prediction = model.predict(feature_vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'
    
if __name__ == "__main__":
    text = input("Enter the text to predict: ")
    print(predict_message(text))