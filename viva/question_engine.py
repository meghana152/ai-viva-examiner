"""
Question Engine Module for AI Viva Examiner.

This module provides the QuestionEngine class to load, filter, and adaptively
select questions from the question bank (data/questions.csv).
"""

import os
import sys
from typing import Dict, List, Optional, Set
import pandas as pd

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class QuestionEngine:
    """
    Manages loading, filtering, and stateful selection of viva examination questions.
    """

    VALID_DIFFICULTIES: List[str] = ["easy", "medium", "hard"]
    REQUIRED_COLUMNS: List[str] = [
        "question_id",
        "subject",
        "topic",
        "difficulty",
        "question",
        "reference_answer",
        "keywords",
    ]

    def __init__(self, questions_path: str = "data/questions.csv"):
        """
        Initialize the QuestionEngine with questions from a CSV file.

        Args:
            questions_path: Relative or absolute path to questions.csv.
        """
        if not os.path.isabs(questions_path):
            questions_path = os.path.join(PROJECT_ROOT, questions_path)

        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions dataset not found at: {questions_path}")

        df = pd.read_csv(questions_path)

        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Questions dataset is missing required columns: {missing_cols}")

        if df.empty:
            raise ValueError("Questions dataset is empty.")

        # Clean string columns
        for col in ["question_id", "subject", "topic", "difficulty", "question"]:
            df[col] = df[col].astype(str).str.strip()

        self.df: pd.DataFrame = df
        self.asked_question_ids: Set[str] = set()

    def _match_topic(self, df_subset: pd.DataFrame, topic: str) -> pd.DataFrame:
        """
        Filter DataFrame by topic using case-insensitive exact or substring matching.

        Args:
            df_subset: DataFrame to filter.
            topic: Topic string (e.g. 'CNN', 'Neural Networks').

        Returns:
            Filtered DataFrame.
        """
        topic_clean = topic.strip().lower()
        exact_matches = df_subset[df_subset["topic"].str.lower() == topic_clean]
        if not exact_matches.empty:
            return exact_matches

        # Substring / Acronym matching (e.g., 'CNN' in 'Convolutional Neural Networks (CNN)')
        return df_subset[df_subset["topic"].str.lower().apply(
            lambda t: f"({topic_clean})" in t or topic_clean in t
        )]

    def get_questions_by_difficulty(self, difficulty: str) -> pd.DataFrame:
        """
        Return all questions matching the requested difficulty level.

        Args:
            difficulty: 'Easy', 'Medium', or 'Hard' (case-insensitive).

        Returns:
            DataFrame containing matching questions.
        """
        diff_clean = difficulty.strip().lower()
        if diff_clean not in self.VALID_DIFFICULTIES:
            raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of Easy, Medium, Hard.")

        return self.df[self.df["difficulty"].str.lower() == diff_clean].copy()

    def get_questions_by_topic(
        self, topic: str, difficulty: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return questions matching the specified topic, optionally filtered by difficulty.

        Args:
            topic: Topic name or abbreviation.
            difficulty: Optional difficulty level to filter by.

        Returns:
            DataFrame containing matching questions.
        """
        if not topic or not isinstance(topic, str) or not topic.strip():
            raise ValueError("Topic must be a non-empty string.")

        filtered = self._match_topic(self.df, topic)

        if difficulty:
            diff_clean = difficulty.strip().lower()
            if diff_clean not in self.VALID_DIFFICULTIES:
                raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of Easy, Medium, Hard.")
            filtered = filtered[filtered["difficulty"].str.lower() == diff_clean]

        return filtered.copy()

    def get_next_question(
        self, difficulty: str, topic: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Select an unasked question matching the requested difficulty and optional topic.

        Fallback behavior:
          If topic is supplied but has no remaining questions for that difficulty,
          falls back to any unasked question matching the requested difficulty.

        Args:
            difficulty: Target difficulty level (Easy, Medium, Hard).
            topic: Optional topic filter.

        Returns:
            Dictionary containing question attributes.

        Raises:
            ValueError: If difficulty is invalid or no unused questions are available.
        """
        diff_clean = difficulty.strip().lower()
        if diff_clean not in self.VALID_DIFFICULTIES:
            raise ValueError(f"Invalid difficulty '{difficulty}'. Must be one of Easy, Medium, Hard.")

        # Exclude questions already asked
        unused_df = self.df[~self.df["question_id"].isin(self.asked_question_ids)].copy()

        # Match difficulty
        diff_unused = unused_df[unused_df["difficulty"].str.lower() == diff_clean]

        candidates = pd.DataFrame()
        if topic and topic.strip():
            # First try matching both topic and difficulty
            topic_diff_unused = self._match_topic(diff_unused, topic)
            if not topic_diff_unused.empty:
                candidates = topic_diff_unused
            else:
                # Fallback: unused questions from requested difficulty without topic filter
                candidates = diff_unused
        else:
            candidates = diff_unused

        if candidates.empty:
            raise ValueError(
                f"No unused questions available for difficulty '{difficulty}'"
                + (f" and topic '{topic}'" if topic else "")
                + "."
            )

        # Deterministic selection: sort by question_id and take the first candidate
        candidates = candidates.sort_values(by="question_id").reset_index(drop=True)
        selected_row = candidates.iloc[0]
        q_id = str(selected_row["question_id"])

        # Mark question as asked
        self.asked_question_ids.add(q_id)

        return {
            "question_id": q_id,
            "subject": str(selected_row["subject"]),
            "topic": str(selected_row["topic"]),
            "difficulty": str(selected_row["difficulty"]),
            "question": str(selected_row["question"]),
            "reference_answer": str(selected_row["reference_answer"]),
            "keywords": str(selected_row["keywords"]),
        }

    def reset(self) -> None:
        """
        Clear the record of asked questions, making all questions available again.
        """
        self.asked_question_ids.clear()

    def get_asked_question_ids(self) -> List[str]:
        """
        Return the list of question IDs that have been asked during the current session.
        """
        return sorted(list(self.asked_question_ids))

    def remaining_questions_count(self) -> int:
        """
        Return the number of unused questions remaining in the question bank.
        """
        return len(self.df) - len(self.asked_question_ids)


def main():
    """
    Test suite for QuestionEngine verifying question counts, filtering,
    adaptive progression, uniqueness, and session resets.
    """
    print("========================================")
    print("QUESTION ENGINE TEST")
    print("========================================")

    engine = QuestionEngine()

    total_q = len(engine.df)
    easy_count = len(engine.get_questions_by_difficulty("Easy"))
    med_count = len(engine.get_questions_by_difficulty("Medium"))
    hard_count = len(engine.get_questions_by_difficulty("Hard"))

    print(f"\nTotal Questions: {total_q}")
    print(f"\nEasy Questions: {easy_count}")
    print(f"Medium Questions: {med_count}")
    print(f"Hard Questions: {hard_count}")

    # Request Easy, Medium, Hard
    q_easy = engine.get_next_question("Easy")
    q_med = engine.get_next_question("Medium")
    q_hard = engine.get_next_question("Hard")

    print("\nSelected Easy Question:")
    print(f"  ID: {q_easy['question_id']} | Topic: {q_easy['topic']} | Diff: {q_easy['difficulty']}")
    print(f"  Question: \"{q_easy['question']}\"")

    print("\nSelected Medium Question:")
    print(f"  ID: {q_med['question_id']} | Topic: {q_med['topic']} | Diff: {q_med['difficulty']}")
    print(f"  Question: \"{q_med['question']}\"")

    print("\nSelected Hard Question:")
    print(f"  ID: {q_hard['question_id']} | Topic: {q_hard['topic']} | Diff: {q_hard['difficulty']}")
    print(f"  Question: \"{q_hard['question']}\"")

    # Verify uniqueness
    selected_ids = [q_easy["question_id"], q_med["question_id"], q_hard["question_id"]]
    assert len(set(selected_ids)) == 3, "Selected questions are not unique!"
    print("\nUnique Selection Check: PASSED")

    # Topic filtering test
    q_cnn_easy = engine.get_next_question(difficulty="Easy", topic="CNN")
    assert "cnn" in q_cnn_easy["topic"].lower() and q_cnn_easy["difficulty"].lower() == "easy"
    print("CNN + Easy Topic Filter: PASSED")

    # Check multiple questions without duplicates
    recent_ids = set(engine.get_asked_question_ids())
    for _ in range(5):
        next_q = engine.get_next_question("Medium")
        assert next_q["question_id"] not in recent_ids
        recent_ids.add(next_q["question_id"])
    print("No Duplicate Questions Check: PASSED")

    # Reset check
    assert engine.remaining_questions_count() < total_q
    engine.reset()
    assert engine.remaining_questions_count() == total_q
    assert len(engine.get_asked_question_ids()) == 0
    print("Reset Check: PASSED")

    print("========================================")


if __name__ == "__main__":
    main()
