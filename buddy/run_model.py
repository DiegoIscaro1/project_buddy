import time
import os
import pandas as pd
import joblib
import glob

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from buddy.preprocessing import transform_input
from sklearn.linear_model import SGDClassifier
from google.cloud import storage

MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

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

# Train the model
def train_model (data, model):
    # Feature/Target
    X = data["text_cleaned"]
    y = data["target"]

    # Fit model
    print("\nFitting Model ...")
    trained_model = model.fit(X,y)
    print("\n✅ Model trained ...")
    return trained_model

# Make predictions
def predict_model (txt: str, trained_model):
    print("Predicting...")
    X_pred = transform_input(txt)
    y_pred = trained_model.predict(X_pred)
    return y_pred

# Save the model in Models folder
def save_model (model, model_name):

    print("\nSaving Model ...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = f'{LOCAL_REGISTRY_PATH}/{model_name}_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print("\n✅ Model saved locally")

    # Save model onn GCS
    if MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{LOCAL_REGISTRY_PATH}/{model_filename}")
        blob.upload_from_filename(model_path)

        print("\n✅ Model saved to GCS")
    else :
        print("\n❌ Model not saved")
    return None

# Load model from models folder
def load_model (model_name):

    if MODEL_TARGET == "gcs":
        print("\nLoading Model on GCS ...")
        client = storage.Client()
        # Get a list of all the blobs in the bucket with the specified prefix
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix=f"{LOCAL_REGISTRY_PATH}/{model_name}_model_"))

        try:
            # Sort the blobs by their creation time and get the latest one
            latest_blob = max(blobs, key=lambda x: x.updated)

            # Download the latest blob to a local file
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name.split("/")[-1])
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = joblib.load(latest_model_path_to_save)

            print("\n✅ Latest model downloaded from cloud storage")

            return latest_model

        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    elif MODEL_TARGET == "local":
        print("\n Loading model locally ...")
        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{LOCAL_REGISTRY_PATH}/{model_name}_model*.pkl")

        if not local_model_paths:
            return None
        most_recent_model_path = sorted(local_model_paths)[-1]

        latest_model = joblib.load(most_recent_model_path)
        print("✅ Model loaded locally...")
        return latest_model
    else:
        return None

# Main to call the function
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
    assert predict_model("I'm super happy",model_loaded) == 0, "\n ❌ Text:'I'm super happy' should be equals to 0 "
    assert predict_model("I wanna kill myself",model_loaded) == 1, "\n ❌ Text: 'I wanna kill myself' should be equals to 1 "
    print ("\n✅ Model does work fine!")
