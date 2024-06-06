
from buddy.run_model import predict_model, load_model
from buddy import preprocessing


def predict(input):
    model_name = "sgd_classifier"
    output = predict_model(input,load_model(model_name))
    return output

if __name__ == "__main__":
    input = "I feel sad. I will kill myself"
    output = predict(input)
    if output == 0:
        result = "This person is going fine!"
    else:
        result = "This person is going to commit SUICIDE!!!"
    print (result)
