def main():

    db_path = "data/inventory.db"

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    X, y = prepare_features(df)

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

    # Select best model (Lowest MAE)
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
    print("\nBest Model:")
    print(best_model_name)

    print(f"MAE: {best_model['mae']:.2f}")
    print(f"RMSE: {best_model['rmse']:.2f}")
    print(f"R2: {best_model['r2']:.2f}%")

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"

    joblib.dump(best_model_obj, model_path)

    print(f"\nBest model saved successfully at: {model_path}")


if __name__ == "__main__":
    main()