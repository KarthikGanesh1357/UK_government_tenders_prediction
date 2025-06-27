import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def main():
    input_path = "C:/nmlops/tender-overshoot-mlops/data/processed/tenders_labeled.csv"
    output_dir = "C:/nmlops/tender-overshoot-mlops/data/processed"

    # Load cleaned data
    df = pd.read_csv(input_path)

    # Fix: Fill missing descriptions (important for TfidfVectorizer)
    df["Description"] = df["Description"].fillna("")

    # Add new feature: delta between estimated and final cost
    df["delta_cost"] = df["final_cost"] - df["estimated_cost"]

    # Features and labels
    X = df[["Description", "Region", "Organisation Name", "estimated_cost", "final_cost", "delta_cost"]]
    y = df["overshot"]

    # Preprocessing pipeline
    preprocess = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=100), "Description"),
        ("region_ohe", OneHotEncoder(handle_unknown="ignore"), ["Region"]),
        ("org_ohe", OneHotEncoder(handle_unknown="ignore"), ["Organisation Name"]),
        ("num", StandardScaler(), ["estimated_cost", "final_cost", "delta_cost"])
    ])

    # Fit and transform features
    X_processed = preprocess.fit_transform(X)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(X_processed, f"{output_dir}/X.pkl")
    joblib.dump(y, f"{output_dir}/y.pkl")
    joblib.dump(preprocess, f"{output_dir}/preprocessor.joblib")

    print("✅ Preprocessing complete. Files saved to 'data/processed/'")


if __name__ == "__main__":
    main()
