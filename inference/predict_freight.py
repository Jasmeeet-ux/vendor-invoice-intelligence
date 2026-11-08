import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"


def load_model(model_path=MODEL_PATH):
    model = joblib.load(model_path)
    return model


def predict_freight_cost(input_data):

    model = load_model()

    input_df = pd.DataFrame(input_data)

    predictions = model.predict(input_df)

    input_df["Predicted_Freight"] = predictions.round(2)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)