import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"


def load_model(model_path=MODEL_PATH):

    model = joblib.load(model_path)

    return model


def predict_invoice_flag(input_data):

    model = load_model()

    input_df = pd.DataFrame(input_data)

    predictions = model.predict(input_df)

    input_df["Predicted_Flag"] = predictions

    return input_df