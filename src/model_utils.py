from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_preparation import load_and_prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def train_evaluate_save_model(model, model_name):
    """
    This function trains one machine learning model,
    evaluates it, saves the trained model,
    and saves the performance results.
    """

    print("\n======================================")
    print("Training model:", model_name)
    print("======================================")

    X_train, X_test, y_train, y_test, preprocessor, metadata = load_and_prepare_data()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("\nFitting the model...")
    pipeline.fit(X_train, y_train)

    print("Making predictions...")
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nModel Performance:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    if roc_auc is not None:
        print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low Interest", "High Interest"],
        zero_division=0
    )
    print(report)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "Reader Interest Threshold": metadata["interest_threshold"],
        "Training Rows": len(X_train),
        "Testing Rows": len(X_test)
    }

    metrics_df = pd.DataFrame([metrics])

    metrics_path = OUTPUT_DIR / f"{model_name}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    report_path = OUTPUT_DIR / f"{model_name}_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Classification Report for {model_name}\n")
        file.write("=" * 60)
        file.write("\n\n")
        file.write(report)

    model_path = MODEL_DIR / f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)

    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Interest", "High Interest"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(
        OUTPUT_DIR / f"{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nSaved files:")
    print("Model:", model_path)
    print("Metrics:", metrics_path)
    print("Classification Report:", report_path)
    print("Confusion Matrix:", OUTPUT_DIR / f"{model_name}_confusion_matrix.png")

    print("\nModel training completed successfully.")

    return metrics