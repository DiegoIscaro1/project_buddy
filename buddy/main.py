from buddy.run_model import model_predicting, load_model


def predict(input: str)->int:
    model_name = "log_reg"
    output = model_predicting(input,load_model(model_name),model_name)
    if output <= 0.25:
        result = 0
    elif output <= 0.6:
        result = 1
    elif output <=0.80:
        result = 2
    else:
        result = 3

    return result

if __name__ == "__main__":
    input = "I feel sad. I will get better if people help me"
    output = predict(input)
    print(output)
