import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():

    # Database path
    db_path = Path("data/inventory.db")

    # Create models directory
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check database exists
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Load data
    df = load_vendor_invoice_data(str(db_path))

    # Prepare data
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = []

    results.append(
        evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    )

    results.append(
        evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression")
    )

    results.append(
        evaluate_model(rf_model, X_test, y_test, "Random Forest Regression")
    )

    # Select best model using lowest MAE
    best_model = min(results, key=lambda x: x["mae"])

    best_model_name = best_model["model_name"]

    # Get best model object
    models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model_obj = models[best_model_name]

    # Display best model
    print("\n========== BEST MODEL ==========")
    print(best_model_name)

    print(f"MAE  : {best_model['mae']:.2f}")
    print(f"RMSE : {best_model['rmse']:.2f}")
    print(f"R2   : {best_model['r2']:.2f}%")

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"

    joblib.dump(best_model_obj, model_path)

    print(f"\nBest model saved successfully at: {model_path}")


if __name__ == "__main__":
    main()