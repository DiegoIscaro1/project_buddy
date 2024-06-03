
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Clean the data
def clean_text(text):
    sentence = sentence.strip()
    sentence = ' '.join(re.split(r'(?<=[a-z])(?=[A-Z])', sentence))
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    sentence = ''.join(char for char in sentence if not char.isdigit())
    tokenized = word_tokenize(sentence) # Tokenize
    stop_words = set(stopwords.words('english')) # Make stopword list
    without_stopwords = [word for word in tokenized if not word in stop_words] # Remove Stop Words
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
    cleaned = ' '.join(lemmatized) # Join back to a string
    return cleaned
