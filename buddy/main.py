from buddy.run_model import model_predicting, load_model


def predict(input: str)->int:
    model_name = "log_reg" # Choice of the model
    output = model_predicting(input,load_model(model_name),model_name)
    if output <= 0.25:
        result = 0
    elif output <= 0.5:
        result = 1
    elif output <= 0.80:
        result = 2
    else:
        result = 3

    return result

if __name__ == "__main__":
    input = "There's a constant weight on my chest, and I can't seem to shake it off."
    result, output = predict(input)
    print(result)
