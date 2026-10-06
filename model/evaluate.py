"""
Model Evaluation Module for AI Viva Examiner.

This script loads the trained Bidirectional LSTM model and fitted tokenizer,
reproduces the grouped test split, calculates evaluation metrics (MAE, MSE, RMSE, R²),
prints sample predictions, and generates visual diagnostic plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow import keras
from preprocessing.text_processor import TextProcessor


def load_and_split_data(
    dataset_path: str, test_size: float = 0.2, random_state: int = 42
) -> pd.DataFrame:
    """
    Load dataset and extract the exact same test partition used during training.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, test_idx = next(gss.split(df, groups=df["question_id"]))

    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    return test_df


def main():
    """
    Execute model evaluation pipeline and plot generation.
    """
    dataset_path = os.path.join(PROJECT_ROOT, "data", "answer_dataset.csv")
    model_path = os.path.join(PROJECT_ROOT, "model", "viva_bilstm.keras")
    tokenizer_path = os.path.join(PROJECT_ROOT, "model", "tokenizer.json")
    actual_vs_pred_path = os.path.join(PROJECT_ROOT, "model", "actual_vs_predicted.png")
    residual_path = os.path.join(PROJECT_ROOT, "model", "residual_plot.png")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train.py first.")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer artifact not found at: {tokenizer_path}. Please run train.py first.")

    # 1. Load test partition
    test_df = load_and_split_data(dataset_path, test_size=0.2, random_state=42)
    test_answers = test_df["student_answer"].tolist()
    y_test = test_df["score"].values.astype(np.float32)

    # 2. Load model and tokenizer
    print("Loading trained model and tokenizer...")
    model = keras.models.load_model(model_path)
    text_processor = TextProcessor(max_vocab_size=5000, max_length=100)
    text_processor.load_tokenizer(tokenizer_path)

    # 3. Preprocess test answers and generate predictions
    X_test = text_processor.transform(test_answers)
    raw_predictions = model.predict(X_test, verbose=0).flatten()

    # 4. Clip predictions to valid 0.0 - 10.0 scale
    y_pred = np.clip(raw_predictions, 0.0, 10.0)

    # 5. Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    num_samples = len(test_df)
    num_questions = test_df["question_id"].nunique()

    # 6. Print evaluation report
    print("\n" + "=" * 40)
    print("AI VIVA EXAMINER - MODEL EVALUATION")
    print("=" * 40)
    print(f"Test Samples: {num_samples}")
    print(f"Test Questions: {num_questions} (Unseen during training)")
    print(f"\nMAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 40)

    # 7. Print prediction range verification
    min_pred = float(np.min(y_pred))
    max_pred = float(np.max(y_pred))
    all_within_range = bool(np.all((y_pred >= 0.0) & (y_pred <= 10.0)))

    print(f"\nMinimum Predicted Value: {min_pred:.2f}")
    print(f"Maximum Predicted Value: {max_pred:.2f}")
    print(f"All Predictions Within [0.0, 10.0]: {all_within_range}")

    # 8. Print sample predictions table (20 rows)
    print("\n" + "=" * 65)
    print(f"{'Question ID':<12} | {'Actual Score':<12} | {'Predicted Score':<15} | {'Absolute Error':<14}")
    print("-" * 65)
    for i in range(20):
        q_id = test_df.iloc[i]["question_id"]
        actual = y_test[i]
        pred = y_pred[i]
        abs_err = abs(actual - pred)
        print(f"{q_id:<12} | {actual:<12.1f} | {pred:<15.2f} | {abs_err:<14.2f}")
    print("=" * 65)

    # 9. Plot 1: Actual vs Predicted Scatter Plot
    print("\nGenerating actual-vs-predicted plot...")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.75, color="#1f77b4", edgecolors="k", s=45, label="Test Predictions")
    plt.plot([0, 10], [0, 10], color="red", linestyle="--", linewidth=1.5, label="y = x (Perfect Fit)")
    plt.title("Actual vs Predicted Viva Scores", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Score", fontsize=11)
    plt.ylabel("Predicted Score", fontsize=11)
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(actual_vs_pred_path, dpi=300)
    plt.close()
    print(f"Saved actual vs predicted plot to: {actual_vs_pred_path}")

    # 10. Plot 2: Residual Plot
    print("Generating residual plot...")
    prediction_errors = y_pred - y_test
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, prediction_errors, alpha=0.75, color="#2ca02c", edgecolors="k", s=45, label="Residuals")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line (e = 0)")
    plt.title("Residual / Prediction Error Plot", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Score", fontsize=11)
    plt.ylabel("Prediction Error", fontsize=11)
    plt.xlim(-0.5, 10.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"Saved residual plot to: {residual_path}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
