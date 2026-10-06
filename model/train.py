"""
Model Training Pipeline for AI Viva Examiner.

This script trains a Bidirectional LSTM neural network to predict viva answer quality
scores (continuous range 0.0 to 10.0) based on student response text.
"""

import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow import keras
from preprocessing.text_processor import TextProcessor


def validate_and_load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Validate dataset existence, schema integrity, and absence of null values.

    Args:
        dataset_path: Path to answer_dataset.csv.

    Returns:
        Loaded pandas DataFrame.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required_columns = [
        "answer_id",
        "question_id",
        "student_answer",
        "reference_answer",
        "score",
        "label",
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if df["student_answer"].isnull().any():
        raise ValueError("Dataset contains null/empty values in 'student_answer'.")

    if df["score"].isnull().any():
        raise ValueError("Dataset contains null/empty values in 'score'.")

    print(f"Dataset loaded and validated successfully: {len(df)} total rows.")
    return df


def split_data_by_question(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a grouped train/test split on question_id to prevent data leakage.
    Answers to the same question will not appear across both train and test partitions.

    Args:
        df: Input DataFrame.
        test_size: Proportion of questions in test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["question_id"]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"Grouped train/test split completed:")
    print(f"  Training samples : {len(train_df)} ({train_df['question_id'].nunique()} unique questions)")
    print(f"  Testing samples  : {len(test_df)} ({test_df['question_id'].nunique()} unique questions)")

    # Assert no question leakage
    overlap = set(train_df["question_id"]).intersection(set(test_df["question_id"]))
    if overlap:
        raise RuntimeError(f"Data leakage detected! Overlapping question IDs: {overlap}")

    return train_df, test_df


def build_bilstm_model(vocab_size: int, max_length: int = 100) -> keras.Model:
    """
    Construct the Bidirectional LSTM regression architecture.

    Pipeline:
      Input (max_length)
      ↓
      Embedding (input_dim=vocab_size + 1, output_dim=64)
      ↓
      Bidirectional(LSTM(64))
      ↓
      Dropout(0.3)
      ↓
      Dense(32, activation='relu')
      ↓
      Dense(1, activation='linear')

    Args:
        vocab_size: Total vocabulary size.
        max_length: Maximum sequence length.

    Returns:
        Compiled Keras Model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(max_length,), name="input_sequence"),
        keras.layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=64,
            name="embedding_layer"
        ),
        keras.layers.Bidirectional(
            keras.layers.LSTM(64),
            name="bidirectional_lstm"
        ),
        keras.layers.Dropout(0.3, name="dropout_layer"),
        keras.layers.Dense(32, activation="relu", name="dense_features"),
        keras.layers.Dense(1, activation="linear", name="score_output")
    ], name="viva_bilstm_evaluator")

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def main():
    """
    Execute end-to-end training and evaluation workflow.
    """
    dataset_path = os.path.join(PROJECT_ROOT, "data", "answer_dataset.csv")
    model_dir = os.path.join(PROJECT_ROOT, "model")
    model_save_path = os.path.join(model_dir, "viva_bilstm.keras")
    tokenizer_save_path = os.path.join(model_dir, "tokenizer.json")

    os.makedirs(model_dir, exist_ok=True)

    print("=" * 80)
    print("AI VIVA EXAMINER - MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load and validate dataset
    df = validate_and_load_dataset(dataset_path)

    # 2. Grouped split by question_id
    train_df, test_df = split_data_by_question(df, test_size=0.2, random_state=42)

    # 3. Fit tokenizer ONLY on training student answers
    print("\nInitializing and fitting TextProcessor on training data...")
    text_processor = TextProcessor(max_vocab_size=5000, max_length=100)
    train_answers = train_df["student_answer"].tolist()
    text_processor.fit(train_answers)

    vocab_size = len(text_processor.tokenizer.word_index)
    print(f"Vocabulary size learned from training set: {vocab_size} tokens")

    # 4. Transform training and testing text
    X_train = text_processor.transform(train_answers)
    X_test = text_processor.transform(test_df["student_answer"].tolist())

    y_train = train_df["score"].values.astype(np.float32)
    y_test = test_df["score"].values.astype(np.float32)

    # 5. Build and inspect model
    print("\nBuilding Bidirectional LSTM regression model...")
    model = build_bilstm_model(vocab_size=vocab_size, max_length=100)
    model.summary()

    # 6. Configure EarlyStopping and train
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1
    )

    # 7. Training summary metrics
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    final_train_mae = history.history["mae"][-1]
    final_val_mae = history.history["val_mae"][-1]

    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Epochs Run        : {len(history.history['loss'])}")
    print(f"Final Training Loss (MSE): {final_train_loss:.4f}")
    print(f"Final Validation Loss    : {final_val_loss:.4f}")
    print(f"Final Training MAE      : {final_train_mae:.4f}")
    print(f"Final Validation MAE    : {final_val_mae:.4f}")

    # 8. Save model and tokenizer
    print(f"\nSaving model to: {model_save_path}")
    model.save(model_save_path)

    print(f"Saving tokenizer to: {tokenizer_save_path}")
    text_processor.save_tokenizer(tokenizer_save_path)

    # 9. Test set evaluation
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE       : {test_mae:.4f}")

    # 10. Sample predictions
    print("\nSample Predictions on Test Set (Clipped 0.0 - 10.0):")
    raw_preds = model.predict(X_test[:10], verbose=0).flatten()
    clipped_preds = np.clip(raw_preds, 0.0, 10.0)

    for idx, (actual, pred) in enumerate(zip(y_test[:10], clipped_preds), 1):
        print(f"  [{idx:2d}] Actual Score: {actual:4.1f} | Predicted Score: {pred:4.2f}")

    print("=" * 80)
    print("Model training and evaluation successfully completed.")


if __name__ == "__main__":
    main()
