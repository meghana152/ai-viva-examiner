"""
Adaptive Question Engine for AI Viva Examiner.

This module determines the difficulty of the next viva question (Easy, Medium, Hard)
based on the student's evaluated answer score.
"""

from typing import List, Union


class AdaptiveEngine:
    """
    Adaptive engine for dynamically adjusting question difficulty.

    Rules:
      - score >= 8.0: Increase difficulty by one level (capped at Hard).
      - 5.0 <= score < 8.0: Maintain current difficulty.
      - score < 5.0: Decrease difficulty by one level (floored at Easy).
    """

    DIFFICULTY_LEVELS: List[str] = ["Easy", "Medium", "Hard"]

    def __init__(self, initial_difficulty: str = "Easy"):
        """
        Initialize the AdaptiveEngine with a starting difficulty level.

        Args:
            initial_difficulty: Initial difficulty level ("Easy", "Medium", or "Hard").
        """
        if initial_difficulty not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid initial difficulty '{initial_difficulty}'. "
                f"Must be one of {self.DIFFICULTY_LEVELS}."
            )
        self.current_difficulty: str = initial_difficulty

    def _validate_score(self, score: Union[int, float]) -> float:
        """
        Validate that the provided score is a valid numeric value within [0.0, 10.0].

        Args:
            score: Input score to validate.

        Returns:
            Validated score as a float.

        Raises:
            ValueError: If score is non-numeric, < 0.0, or > 10.0.
        """
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(
                f"Score must be a numeric value, got {type(score).__name__}."
            )
        if score < 0.0 or score > 10.0:
            raise ValueError(
                f"Invalid score {score}. Score must be between 0.0 and 10.0."
            )
        return float(score)

    def get_difficulty(self) -> str:
        """
        Return the current question difficulty level.

        Returns:
            Current difficulty ("Easy", "Medium", or "Hard").
        """
        return self.current_difficulty

    def reset(self, initial_difficulty: str = "Easy") -> None:
        """
        Reset the engine to the specified starting difficulty.

        Args:
            initial_difficulty: Target difficulty to reset to (default "Easy").
        """
        if initial_difficulty not in self.DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid difficulty '{initial_difficulty}'. "
                f"Must be one of {self.DIFFICULTY_LEVELS}."
            )
        self.current_difficulty = initial_difficulty

    def get_adaptation_reason(self, score: Union[int, float]) -> str:
        """
        Provide a human-readable explanation for difficulty adjustment.

        Args:
            score: Evaluated score (0.0 - 10.0).

        Returns:
            Reason string explaining why difficulty increased, maintained, or decreased.
        """
        score = self._validate_score(score)

        if score >= 8.0:
            if self.current_difficulty == "Hard":
                return "Strong answer (score >= 8): maintaining maximum difficulty (Hard)."
            return "Strong answer (score >= 8): increasing difficulty."
        elif score >= 5.0:
            return "Moderate answer (5 <= score < 8): maintaining difficulty."
        else:
            if self.current_difficulty == "Easy":
                return "Weak answer (score < 5): maintaining minimum difficulty (Easy)."
            return "Weak answer (score < 5): decreasing difficulty."

    def update_difficulty(self, score: Union[int, float]) -> str:
        """
        Update the current difficulty level based on the evaluation score.

        Transitions:
          - Easy   + (>= 8) -> Medium
          - Easy   + (< 5)  -> Easy
          - Medium + (>= 8) -> Hard
          - Medium + (< 5)  -> Easy
          - Hard   + (>= 8) -> Hard
          - Hard   + (< 5)  -> Medium
          - Any    + (5..8) -> Same

        Args:
            score: Student's evaluation score (0.0 to 10.0).

        Returns:
            The new difficulty level.
        """
        score = self._validate_score(score)

        if score >= 8.0:
            # Increase difficulty by one level
            if self.current_difficulty == "Easy":
                self.current_difficulty = "Medium"
            elif self.current_difficulty == "Medium":
                self.current_difficulty = "Hard"
            elif self.current_difficulty == "Hard":
                self.current_difficulty = "Hard"
        elif score >= 5.0:
            # Maintain current difficulty
            pass
        else:
            # Decrease difficulty by one level
            if self.current_difficulty == "Hard":
                self.current_difficulty = "Medium"
            elif self.current_difficulty == "Medium":
                self.current_difficulty = "Easy"
            elif self.current_difficulty == "Easy":
                self.current_difficulty = "Easy"

        return self.current_difficulty


def main():
    """
    Test suite for AdaptiveEngine covering all required state transitions and edge cases.
    """
    print("========================================")
    print("ADAPTIVE ENGINE TEST")
    print("========================================")

    test_scenarios = [
        ("Easy", 9.0, "Medium", "1. Easy + score 9 -> Medium"),
        ("Medium", 9.0, "Hard", "2. Medium + score 9 -> Hard"),
        ("Hard", 9.0, "Hard", "3. Hard + score 9 -> Hard"),
        ("Hard", 4.0, "Medium", "4. Hard + score 4 -> Medium"),
        ("Medium", 4.0, "Easy", "5. Medium + score 4 -> Easy"),
        ("Easy", 4.0, "Easy", "6. Easy + score 4 -> Easy"),
        ("Medium", 6.0, "Medium", "7. Medium + score 6 -> Medium"),
    ]

    engine = AdaptiveEngine()

    for start_diff, score, expected, desc in test_scenarios:
        engine.reset(start_diff)
        initial = engine.get_difficulty()
        reason = engine.get_adaptation_reason(score)
        next_diff = engine.update_difficulty(score)

        print(f"\nScenario: {desc}")
        print(f"Initial Difficulty: {initial}")
        print(f"Score: {score:.1f}")
        print(f"Next Difficulty: {next_diff}")
        print(f"Reason: {reason}")

        assert next_diff == expected, f"Failed: expected {expected}, got {next_diff}"

    # Invalid score tests
    print("\n----------------------------------------")
    print("VALIDATING INVALID SCORE HANDLING")
    print("----------------------------------------")

    invalid_scores = [-1, 11, "high", None]
    for inv in invalid_scores:
        try:
            engine.update_difficulty(inv)  # type: ignore
            print(f"ERROR: Invalid score {inv} did not raise ValueError!")
        except ValueError as e:
            print(f"Verified: Invalid score {inv} raised ValueError -> '{e}'")

    print("\n========================================")
    print("All AdaptiveEngine test assertions passed.")
    print("========================================")


if __name__ == "__main__":
    main()
