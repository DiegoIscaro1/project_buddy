import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from transformers import BertTokenizer
from tqdm.auto import tqdm

tqdm.pandas()

def label_encoding_target (data: pd.DataFrame):
    # Initialize Label encoder
    le = preprocessing.LabelEncoder()
    # Encode the target variable and store it in a new column named "target"
    data["target"] =  le.fit_transform(data["class"])
    # Drop old column
    data = data.drop(columns="class")
    # return new data
    return data

# Function to clean the data
def clean_text(sentence):
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
    # Tokenize the text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized = tokenizer.tokenize(sentence)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    without_stopwords = [word for word in tokenized if not word in stop_words]
    # Initiate Lemmatizer
    lemma=WordNetLemmatizer()
    # Lemmatize
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords]
    # Join the lemmatized words back into a string
    cleaned = ' '.join(lemmatized)
    # Return clean data
    return cleaned

# Function to upload cleaned data to a CSV file
def upload_csv (data: pd.DataFrame):
    data.to_csv("raw_data/Suicide_Detection_cleaned.csv", index=False)

def preprocess_data(data: pd.DataFrame):
    print("Text Cleaning...")
    # Apply the clean_text function to the "text" column and store the result in a new column named "text_cleaned"
    data.loc[:, "text_cleaned"] = data["text"].progress_map(clean_text)
    print ("Cleaning Done!")
    # Replace empty strings with NaN values
    data['text_cleaned'] = data['text_cleaned'].map(lambda x: np.nan if x == '' else x)
    # Drop rows with NaN values in the "text_cleaned" column and remove duplicates
    data = data.dropna(axis=0).drop_duplicates(subset=['text_cleaned'])
    # Drop the original "text" column
    data = data.drop(columns="text")
    # Encode the target variable
    data = label_encoding_target(data)
    # Upload the cleaned data to a CSV file
    upload_csv(data)
    return data

# Function to transform input text
def transform_input (text: str):
    '''transform the input we received from the frontend
    into something we can process'''
    cleaned_text = clean_text(text)
    return pd.Series(cleaned_text)

if __name__ == "__main__":
    data = pd.read_csv("raw_data/Suicide_Detection.csv",index_col=0)
    data = preprocess_data(data)
