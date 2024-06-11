import time
import os
import pandas as pd
import joblib
import glob

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from buddy.preprocessing import transform_input
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from google.cloud import storage

MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")

def logreg() -> Pipeline:
    # Pipeline vectorizer + Logreg
    pipeline_log_reg = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=4000)),
        ('model', LogisticRegression(C=1, penalty='l1', solver='liblinear')),
    ])
    return pipeline_log_reg

def naive_bayes_model() -> Pipeline:
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(max_features=4000,ngram_range=(1,2)),
        MultinomialNB(alpha=1)
    )
    return pipeline_naive_bayes

def sgd_model() -> Pipeline:
    # Pipeline vectorizer + sgd classifier
    pipeline_sgd = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000)),
        ('classifier', SGDClassifier(alpha=0.0001, loss='hinge', penalty='l2')),
    ])
    return pipeline_sgd

# Model choice
def model_choice (model_name) -> Pipeline:
    if model_name == "sgd_classifier":
        model = sgd_model()
        print(f"\n✅ Model {model_name} found ...")
        return model
    elif model_name == "naive_bayes":
        model = naive_bayes_model()
        print(f"\n✅ Model {model_name} found ...")
        return model
    elif model_name == "log_reg":
        model = logreg()
        print(f"\n✅ Model {model_name} found ...")
        return model
    else:
        print("\n❌ No model of that name. Please find an existing model")


# Evaluate model
def evaluate_model (data: pd.DataFrame, model: Pipeline) -> float:
    print ("\nStarting to evaluate the model ...")
    X = data["text_cleaned"]
    y = data["target"]

    # Cross-validation
    cv_results = cross_validate(model,
                                X,
                                y,
                                cv = 5,
                                scoring = ["accuracy"],
                                verbose=2)
    mean_accuracy = cv_results["test_accuracy"].mean()
    print (f"\nEvaluation : The model is accurate to {round(mean_accuracy,4)}")
    return mean_accuracy

# Train the model
def train_model (data: pd.DataFrame, model: Pipeline) -> Pipeline:
    # Feature/Target
    X = data["text_cleaned"]
    y = data["target"]

    # Fit model
    print("\nFitting Model ...")
    trained_model = model.fit(X,y)
    print("\n✅ Model trained ...")
    return trained_model

# Make predictions
def model_predicting (txt: str, trained_model: Pipeline, model_name: str) -> float:
    print("\nPredicting...")
    X_pred = transform_input(txt)

    # Give the probability of being 1
    if model_name == "log_reg":
        y_prob = trained_model.predict_proba(X_pred)
        y_prob_suicide = round(y_prob[0,1],4)
        return y_prob_suicide
    # Give the prediction
    else:
        y_pred = trained_model.predict(X_pred)
        y_pred_result = y_pred[0]
        return y_pred_result

# Save the model in Models folder
def save_model (model: Pipeline, model_name: str):

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
def load_model (model_name: str) -> Pipeline:

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

def running_model(model_name: str):

    # Initiate model
    model = model_choice(model_name)

    # Load data
    data = pd.read_csv("raw_data/Suicide_Detection_cleaned.csv")

    # Evaluate model
    accuracy = evaluate_model(data,model)

    # Train model if sufficient accuracy
    if accuracy > 0.9:
        trained_model = train_model(data,model)

        # Save model
        save_model(trained_model, model_name)

        # Load model
        model_loaded = load_model(model_name)

        # Test model
        y_pred = model_predicting("I'm super happy",model_loaded,model_name)
        assert y_pred <= 0.5, print("\n ❌ Text:'I'm super happy' should be equals to 0 ")
        y_pred = model_predicting("I wanna kill myself",model_loaded,model_name)
        assert y_pred >= 0.5, print("\n ❌ Text: 'I wanna kill myself' should be equals to 1 ")
        print ("\n✅ Model does work fine!")


    # If not suffucient accuracy, drop the model
    else :
        print ("\n ❌ Model accuracy is too low to be saved. \n Please fined-tune the model")

# Main to call the function
if __name__ == "__main__":

    # Initiate model
    model_name = "log_reg"
    running_model(model_name)
