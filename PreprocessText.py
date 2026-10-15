import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, mode='tokenized'):
    tokens = word_tokenize(str(text).lower())
    
    processed_tokens = []
    for token in tokens:
        if token in stop_words:
            continue
            
        if mode == 'stemmed':
            processed_tokens.append(stemmer.stem(token))
        elif mode == 'lemmatized':
            processed_tokens.append(lemmatizer.lemmatize(token))
        else:
            processed_tokens.append(token)
            
    return " ".join(processed_tokens)

