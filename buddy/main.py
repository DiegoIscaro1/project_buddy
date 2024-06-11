from buddy.run_model import model_predicting, load_model


def predict(input: str)->float:
    model_name = "log_reg"
    output = model_predicting(input,load_model(model_name),model_name)
    return output

if __name__ == "__main__":
    input = "I feel sad. I will get better if people help me"
    output = predict(input)
    print(output)
