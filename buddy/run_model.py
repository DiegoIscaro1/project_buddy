import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from buddy.preprocessing import transform_input
import joblib
from sklearn.linear_model import SGDClassifier

def naive_bayes_model():
    # Pipeline vectorizer + Naive Bayes
    model_name = "naive_bayes"
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(max_features=4000,ngram_range=(1,2)),
        MultinomialNB(alpha=1)
    )
    return pipeline_naive_bayes, model_name

def sgd_model():
    # Pipeline vectorizer + sgd
    model_name = "sgd_classifier"
    pipeline_sgd = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', SGDClassifier(alpha=0.0001, loss='hinge', penalty='l2')),
    ])
    return pipeline_sgd,model_name

def train_model (data, model):
    # Feature/Target
    X = data["text_cleaned"]
    y = data["target"]

    # Fit model
    print("Fitting Model ...")
    trained_model = model.fit(X,y)
    print("Model trained ...")
    return trained_model

def predict_model (txt: str, trained_model):
    print("Predicting...")
    X_pred = transform_input(txt)
    y_pred = trained_model.predict(X_pred)
    return y_pred

def save_model (model, model_name):
    print("Saving Model ...")
    joblib.dump(model, f'models/{model_name}_model.pkl')

def load_model (model_name):
    print("Loading Model ...")
    model = joblib.load(f"models/{model_name}_model.pkl")
    print("Model loaded...")
    return model

if __name__ == "__main__":
    model, model_name = sgd_model()
    save_model(
        train_model(
            pd.read_csv("raw_data/Suicide_Detection_cleaned.csv"),
            model
            ),
        model_name
        )
    model_loaded = load_model(model_name)
    assert predict_model("I'm super happy",model_loaded) == 0, " Text:'I'm super happy' should be equals to 0 "
    assert predict_model("I wanna kill myself",model_loaded) == 1, " Text: 'I wanna kill myself' should be equals to 1 "
    print ("Model does work fine!")
