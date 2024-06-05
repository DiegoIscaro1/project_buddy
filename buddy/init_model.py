import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from buddy.preprocessing import transform_input
import joblib

def naive_bayes_model():
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(max_features=4000,ngram_range=(1,2)),
        MultinomialNB(alpha=1)
    )
    return pipeline_naive_bayes

def train_model (data, model):
    # Feature/Target
    X = data["text_cleaned"]
    y = data["target"]

    # Fit model
    print("Fitting Model ...")
    trained_model = model.fit(X,y)
    print("Model trained ...")
    return trained_model

def predict_model (X_pred, trained_model):
    y_pred = trained_model.predict(X_pred)
    return y_pred

def save_model (model):
    print("Saving Model ...")
    joblib.dump(model, 'models/nlp_model.pkl')

def load_model ():
    print("Loading Model ...")
    model = joblib.load('models/nlp_model.pkl')
    return model

if __name__ == "__main__":
    save_model(
        train_model(
            pd.read_csv("raw_data/Suicide_Detection_cleaned.csv"),
            naive_bayes_model()
            )
        )
