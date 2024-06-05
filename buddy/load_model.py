import joblib

def load_model (model_name):
    print("Loading Model ...")
    model = joblib.load(f"../models/{model_name}_model.pkl")
    print("Model loaded...")
    return model
