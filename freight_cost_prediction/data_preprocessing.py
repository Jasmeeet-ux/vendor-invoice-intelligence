import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split


# 1. Load vendor invoice data
def load_vendor_invoice_data(db_path: str):
    """Load vendor invoice data from SQLite database."""

    conn = sqlite3.connect(db_path)

    try:
        query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    return df


# 2. Prepare features and target
def prepare_features(df: pd.DataFrame):
    """Select features and target variable."""

    X = df[["Dollars"]]
    y = df["Freight"]

    return X, y


# 3. Split dataset into training and testing
def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )