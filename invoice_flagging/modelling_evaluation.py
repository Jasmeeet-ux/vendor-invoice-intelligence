
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score
)


# ==========================================
# 1. TRAIN RANDOM FOREST WITH GRID SEARCH
# ==========================================

def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # Hyperparameter combinations
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best F1 Score:", grid_search.best_score_)

    return grid_search


# ==========================================
# 2. EVALUATE CLASSIFICATION MODEL
# ==========================================

def evaluate_classifier(model, X_test, y_test, model_name):

    # Predict test data
    y_pred = model.predict(X_test)

    print(f"\n{model_name} Performance:")
    print("--------------------------------")

    # Accuracy
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
