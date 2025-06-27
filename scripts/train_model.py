import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # Load preprocessed features and labels
    X = joblib.load("C:/nmlops/tender-overshoot-mlops/data/processed/X.pkl")
    y = joblib.load("C:/nmlops/tender-overshoot-mlops/data/processed/y.pkl")

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Start MLflow experiment
    mlflow.set_experiment("tender_overshoot_detection")
    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log key metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_1", report["1"]["precision"])
        mlflow.log_metric("recall_1", report["1"]["recall"])
        mlflow.log_metric("f1_1", report["1"]["f1-score"])

        # Save model locally
        model_path = "models/random_forest_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)

        # Log model and preprocessor to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("C:/nmlops/tender-overshoot-mlops/data/processed/preprocessor.joblib")

        print("✅ Random Forest model trained and logged with MLflow.")


if __name__ == "__main__":
    main()
