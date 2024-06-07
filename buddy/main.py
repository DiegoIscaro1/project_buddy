
from buddy.run_model import predict_model, load_model
from buddy import preprocessing


def predict(input):
    model_name = "log_reg"
    output = predict_model(input,load_model(model_name),model_name)
    if output <= 0.3:
        result = "This person is going fine!"
    elif output <= 0.7:
        result = "There are some concerns but it is still ok"
    else:
        result = "This person is going to commit SUICIDE!!!"
    return result

if __name__ == "__main__":
    input = "I feel sad. I will get better if people help me"
    output = predict(input)
    print (output)
