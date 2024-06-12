import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm

tqdm.pandas()
# Download the punkt tokenizer (if not already downloaded)
nltk.download("punkt")

def label_encoding_target (data: pd.DataFrame) -> pd.DataFrame:
    # Initialize Label encoder
    le = preprocessing.LabelEncoder()
    # Encode the target variable and store it in a new column named "target"
    data["target"] =  le.fit_transform(data["class"])
    data = data.drop(columns="class")
    return data

def create_stop_words ():
    '''Define the list of Stop Words to be removed'''

    negative = set(["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"])
    reflective_pronouns = set([
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves"
    ])
    stop_words = set(stopwords.words('english'))

    return stop_words - negative - reflective_pronouns

def clean_text(sentence: str, stop_words, lemmatizer) -> str:
    # Remove leading and trailing whitespaces
    sentence = sentence.strip()
    # Split camel case words
    sentence = ' '.join(re.split(r'(?<=[a-z])(?=[A-Z])', sentence))
    # Lower the capitalized letters
    sentence = sentence.lower()
     # Remove punctuation
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    # Remove digits
    sentence = ''.join(char for char in sentence if not char.isdigit())
    # Tokenize the text
    tokenized = word_tokenize(sentence)
    # Remove Stop Words
    without_stopwords = [word for word in tokenized if not word in stop_words]
    # Initiate Lemmatizer
    lemmatized = [lemmatizer.lemmatize(word) for word in without_stopwords]
    # Join the lemmatized words back into a string
    cleaned = ' '.join(lemmatized)
    return cleaned

# Function to upload cleaned data to a CSV file
def upload_csv (data: pd.DataFrame):
    data.to_csv("raw_data/Suicide_Detection_cleaned.csv", index=False)

def preprocess_data(data: pd.DataFrame):
    print("Text Cleaning...")
    stop_words = create_stop_words()
    lemma = WordNetLemmatizer()
    # Apply the clean_text function to the "text" column
    data.loc[:, "text_cleaned"] = data["text"].progress_map(lambda x: clean_text(x,stop_words,lemma))
    print ("Cleaning Done!")
    # Create a boolean mask for the rows where "text_cleaned" is empty
    mask = data["text_cleaned"] == ""
    # Modify the "text_cleaned" column in place for the rows where the mask is True
    data.loc[mask, "text_cleaned"] = np.nan
    # Drop rows with NaN values in the "text_cleaned" column and remove duplicates
    data = data.dropna(axis=0).drop_duplicates(subset=['text_cleaned'])
    data = data.drop(columns="text")
    data = data.reset_index(drop=True)
    # Encode the target variable
    data = label_encoding_target(data)
    upload_csv(data)

# Function to transform input text
def transform_input (text: str) -> pd.Series:
    '''transform the input we received from the frontend
    into something we can process'''
    stop_words = create_stop_words()
    lemma = WordNetLemmatizer()
    cleaned_text = clean_text(text,stop_words,lemma)
    return pd.Series(cleaned_text)

if __name__ == "__main__":
    data = pd.read_csv("raw_data/Suicide_Detection.csv",index_col=0)
    data = preprocess_data(data)
