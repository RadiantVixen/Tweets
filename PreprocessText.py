import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_tweet(text):
    """Basic tweet cleaning: remove URLs, mentions, hashtags, special chars."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text, mode='tokenized'):
    text   = clean_tweet(text)
    tokens = word_tokenize(text)

    processed = []
    for token in tokens:
        if token in stop_words or not token.strip():
            continue
        if mode == 'stemmed':
            processed.append(stemmer.stem(token))
        elif mode == 'lemmatized':
            processed.append(lemmatizer.lemmatize(token))
        else:
            processed.append(token)

    return " ".join(processed)
