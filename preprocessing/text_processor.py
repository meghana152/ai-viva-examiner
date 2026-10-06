"""
Text Preprocessing Module for AI Viva Examiner.

This module provides the TextProcessor class for cleaning and tokenizing
student and reference answers for deep learning models (BiLSTM) using
TensorFlow/Keras utilities.
"""

import os
import pickle
import re
from typing import List, Optional, Union
import numpy as np

# Use TensorFlow Keras preprocessing utilities
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextProcessor:
    """
    NLP Text Processor for tokenization and sequence padding.

    Key characteristics:
    - Retains technical vocabulary without aggressive stopword removal.
    - Avoids stemming/lemmatization to preserve exact semantic terms.
    - Fits tokenizer strictly on training data splits.
    """

    def __init__(
        self,
        max_vocab_size: int = 5000,
        max_length: int = 100,
        oov_token: str = "<OOV>",
        padding: str = "post",
        truncating: str = "post",
    ):
        """
        Initialize TextProcessor configuration.

        Args:
            max_vocab_size: Maximum number of words in vocabulary.
            max_length: Maximum sequence length for padding.
            oov_token: Token used for out-of-vocabulary words.
            padding: Padding direction ('post' or 'pre').
            truncating: Truncation direction ('post' or 'pre').
        """
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        self.tokenizer: Optional[Tokenizer] = None

    def clean_text(self, text: Optional[str]) -> str:
        """
        Clean raw text:
        - Convert text to lowercase.
        - Replace unnecessary punctuation with spaces.
        - Normalize consecutive whitespaces.
        - Safely handle empty, None, or non-string inputs.

        Args:
            text: Input raw string or None.

        Returns:
            Cleaned and normalized string.
        """
        if text is None or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace punctuation characters with space to prevent words from merging
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace (collapse multiple spaces, tabs, newlines) and strip
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def fit(self, texts: List[str]) -> None:
        """
        Fit the Keras Tokenizer on training texts only.
        Should never be fitted on test/validation data.

        Args:
            texts: List of raw or cleaned training text strings.
        """
        if not texts:
            raise ValueError("Input texts list for fit() cannot be empty.")

        cleaned_texts = [self.clean_text(t) for t in texts]

        # Initialize Keras Tokenizer with specified vocabulary limit and OOV token
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token=self.oov_token,
            filters="",   # Punctuation handled by clean_text
            lower=False   # Lowercasing handled by clean_text
        )

        self.tokenizer.fit_on_texts(cleaned_texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts into integer sequences and pad to max_length.

        Args:
            texts: List of text strings to transform.

        Returns:
            2D NumPy array of padded integer sequences (shape: [num_texts, max_length]).
        """
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer has not been fitted or loaded yet. "
                "Call fit() on training data or load_tokenizer() before calling transform()."
            )

        cleaned_texts = [self.clean_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding=self.padding,
            truncating=self.truncating,
        )

        return np.array(padded_sequences, dtype=np.int32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit tokenizer on training texts and return padded sequences.

        Args:
            texts: List of training text strings.

        Returns:
            2D NumPy array of padded integer sequences.
        """
        self.fit(texts)
        return self.transform(texts)

    def save_tokenizer(self, path: str) -> None:
        """
        Serialize and save the fitted tokenizer to disk.

        Args:
            path: Target file path (e.g., 'model/tokenizer.pkl').
        """
        if self.tokenizer is None:
            raise ValueError("Cannot save an unfitted tokenizer. Call fit() first.")

        # Ensure target directory exists
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_tokenizer(self, path: str) -> None:
        """
        Load a previously saved tokenizer from disk.

        Args:
            path: Source file path where tokenizer is saved.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at: {path}")

        with open(path, "rb") as f:
            self.tokenizer = pickle.load(f)
