"""
Answer Evaluator Module for AI Viva Examiner.

This module provides the AnswerEvaluator class to evaluate student viva responses
using the pre-trained Bidirectional LSTM model and fitted tokenizer.
"""

import os
import sys
from typing import Dict, Union, Optional
import numpy as np

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow import keras
from preprocessing.text_processor import TextProcessor


class AnswerEvaluator:
    """
    Evaluates student viva answers using the trained BiLSTM model
    and text preprocessor.
    """

    def __init__(
        self,
        model_path: str = "model/viva_bilstm.keras",
        tokenizer_path: str = "model/tokenizer.json",
    ):
        """
        Initialize the AnswerEvaluator by loading model weights and tokenizer.

        Args:
            model_path: Relative or absolute path to trained .keras model.
            tokenizer_path: Relative or absolute path to tokenizer artifact.
        """
        # Resolve relative paths relative to PROJECT_ROOT
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(PROJECT_ROOT, tokenizer_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer artifact not found at: {tokenizer_path}")

        # 1. Load trained TensorFlow/Keras model
        self.model = keras.models.load_model(model_path)

        # 2. Create TextProcessor and load saved tokenizer
        self.text_processor = TextProcessor(max_vocab_size=5000, max_length=100)
        self.text_processor.load_tokenizer(tokenizer_path)

    def get_category(self, score: float) -> str:
        """
        Determine qualitative evaluation category based on numerical score.

        Thresholds:
            score < 3.0:      "Poor"
            3.0 <= score < 5:  "Weak"
            5.0 <= score < 7:  "Partial"
            7.0 <= score < 9:  "Good"
            score >= 9.0:      "Excellent"

        Args:
            score: Evaluated numerical score in range [0.0, 10.0].

        Returns:
            Category name string.
        """
        if score < 3.0:
            return "Poor"
        elif score < 5.0:
            return "Weak"
        elif score < 7.0:
            return "Partial"
        elif score < 9.0:
            return "Good"
        else:
            return "Excellent"

    def evaluate_answer(self, answer: str) -> Dict[str, Union[float, str]]:
        """
        Clean, tokenize, and evaluate a student's viva answer.

        Args:
            answer: Raw student response text.

        Returns:
            Dictionary containing 'score' (float) and 'category' (str).

        Raises:
            ValueError: If answer is None or contains only whitespace.
        """
        # 1. Input validation
        if answer is None or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Student answer cannot be empty or whitespace-only.")

        # 2. Clean and transform answer using existing TextProcessor
        padded_sequence = self.text_processor.transform([answer])

        # 3. Predict quality score using trained BiLSTM
        raw_prediction = self.model.predict(padded_sequence, verbose=0).flatten()[0]

        # 4. Convert to Python float and clip to [0.0, 10.0]
        score = float(np.clip(raw_prediction, 0.0, 10.0))

        # 5. Determine category
        category = self.get_category(score)

        return {
            "score": round(score, 2),
            "category": category,
        }


def main():
    """
    Test and demonstrate AnswerEvaluator with example answers and error handling.
    """
    print("========================================")
    print("ANSWER EVALUATOR TEST")
    print("========================================")

    evaluator = AnswerEvaluator()

    test_cases = [
        (
            "Answer 1",
            "Neural networks consist of interconnected neurons and learn patterns from training data by adjusting weights.",
        ),
        (
            "Answer 2",
            "CNNs use convolutional layers to extract spatial features from images.",
        ),
        (
            "Answer 3",
            "Backpropagation calculates the error and updates network weights using gradients.",
        ),
    ]

    for label, text in test_cases:
        result = evaluator.evaluate_answer(text)
        print(f"\n{label}")
        print(f"Predicted Score: {result['score']}/10")
        print(f"Category: {result['category']}")

    # Test empty answer validation
    print("\nValidating empty input handling:")
    try:
        evaluator.evaluate_answer("   ")
        print("Failure: Empty answer did not trigger ValueError.")
    except ValueError as e:
        print(f"Verified: ValueError raised as expected -> '{e}'")

    print("========================================")


if __name__ == "__main__":
    main()
