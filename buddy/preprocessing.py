import pandas as pd
import re
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from tqdm.auto import tqdm

tqdm.pandas()

def label_encoding_target (data: pd.DataFrame):
    le = preprocessing.LabelEncoder()
    data["target"] =  le.fit_transform(data["class"])
    data = data.drop(columns="class")
    return data

def clean_text(sentence):
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

def upload_csv (data: pd.DataFrame):
    data.to_csv("raw_data/Suicide_Detection_cleaned.csv", index=False)

def preprocess_data (data: pd.DataFrame):
    print("Text Cleaning...")
    data.loc[:, "text_cleaned"] = data["text"].progress_map(clean_text)
    print ("Cleaning Done!")
    data = data.dropna(axis=0).drop_duplicates(subset=['text_cleaned'])
    data = data.drop(columns="text")
    data = data.reset_index(drop=True)
    data = label_encoding_target(data)
    upload_csv(data)

if __name__ == "__main__":
    data = pd.read_csv("raw_data/Suicide_Detection.csv",index_col=0)
    data = preprocess_data(data)
