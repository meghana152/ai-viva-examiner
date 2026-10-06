"""
AI Viva Examiner - Streamlit Web Application.

An adaptive deep learning viva examination system that evaluates student text responses,
adjusts question difficulty dynamically using a BiLSTM model, and provides a comprehensive
performance evaluation report.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from viva.question_engine import QuestionEngine
from viva.evaluator import AnswerEvaluator
from viva.adaptive_engine import AdaptiveEngine


def initialize_session_state() -> None:
    """
    Initialize persistent Streamlit session state variables and reusable engines.
    Engines are loaded only once to avoid reloading model weights on each rerun.
    """
    if "question_engine" not in st.session_state:
        st.session_state.question_engine = QuestionEngine()

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = AnswerEvaluator()

    if "adaptive_engine" not in st.session_state:
        st.session_state.adaptive_engine = AdaptiveEngine()

    # Viva flow states
    if "viva_started" not in st.session_state:
        st.session_state.viva_started = False

    if "viva_completed" not in st.session_state:
        st.session_state.viva_completed = False

    if "student_name" not in st.session_state:
        st.session_state.student_name = ""

    if "subject" not in st.session_state:
        st.session_state.subject = "Deep Learning"

    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = "All Topics"

    if "current_difficulty" not in st.session_state:
        st.session_state.current_difficulty = "Easy"

    if "question_number" not in st.session_state:
        st.session_state.question_number = 1

    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 5

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "scores" not in st.session_state:
        st.session_state.scores = []

    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False

    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None


def start_viva(
    student_name: str,
    subject: str,
    selected_topic: str,
    initial_difficulty: str,
    total_questions: int,
) -> None:
    """
    Validate student details, reset engines, and fetch the first viva question.
    """
    if not student_name.strip():
        st.sidebar.warning("Please enter your Student Name before starting.")
        return

    if not subject.strip():
        st.sidebar.warning("Please enter a Subject before starting.")
        return

    try:
        # Reset question bank and adaptive difficulty
        st.session_state.question_engine.reset()
        st.session_state.adaptive_engine.reset(initial_difficulty=initial_difficulty)

        st.session_state.student_name = student_name.strip()
        st.session_state.subject = subject.strip()
        st.session_state.selected_topic = selected_topic
        st.session_state.current_difficulty = initial_difficulty
        st.session_state.question_number = 1
        st.session_state.total_questions = int(total_questions)
        st.session_state.scores = []
        st.session_state.answer_submitted = False
        st.session_state.last_evaluation = None
        st.session_state.viva_completed = False

        # Load first question
        topic_filter = None if selected_topic == "All Topics" else selected_topic
        first_q = st.session_state.question_engine.get_next_question(
            difficulty=initial_difficulty, topic=topic_filter
        )
        st.session_state.current_question = first_q
        st.session_state.viva_started = True

    except Exception as e:
        st.sidebar.error(f"Error starting viva examination: {e}")


def submit_answer(student_answer: str) -> None:
    """
    Evaluate the student's answer using AnswerEvaluator and update AdaptiveEngine.
    """
    if not student_answer.strip():
        st.warning("Please write an answer before submitting.")
        return

    if st.session_state.answer_submitted:
        # Prevent duplicate submissions
        return

    try:
        current_q = st.session_state.current_question
        eval_result = st.session_state.evaluator.evaluate_answer(student_answer)

        # Update difficulty in adaptive engine
        next_diff = st.session_state.adaptive_engine.update_difficulty(
            eval_result["score"]
        )

        # Record record for final report
        record = {
            "question_number": st.session_state.question_number,
            "question_id": current_q["question_id"],
            "topic": current_q["topic"],
            "difficulty": current_q["difficulty"],
            "question": current_q["question"],
            "student_answer": student_answer.strip(),
            "score": eval_result["score"],
            "category": eval_result["category"],
            "next_difficulty": next_diff,
        }
        st.session_state.scores.append(record)

        st.session_state.last_evaluation = {
            "score": eval_result["score"],
            "category": eval_result["category"],
            "next_difficulty": next_diff,
        }
        st.session_state.current_difficulty = next_diff
        st.session_state.answer_submitted = True

    except Exception as e:
        st.error(f"Evaluation encountered an error: {e}")


def load_next_question() -> None:
    """
    Advance to the next question or transition to the final report if finished.
    """
    if st.session_state.question_number >= st.session_state.total_questions:
        st.session_state.viva_completed = True
        return

    try:
        st.session_state.question_number += 1
        st.session_state.answer_submitted = False
        st.session_state.last_evaluation = None

        topic_filter = (
            None
            if st.session_state.selected_topic == "All Topics"
            else st.session_state.selected_topic
        )

        next_q = st.session_state.question_engine.get_next_question(
            difficulty=st.session_state.current_difficulty, topic=topic_filter
        )
        st.session_state.current_question = next_q

    except Exception as e:
        st.error(f"Error loading next question: {e}")


def show_viva_screen() -> None:
    """
    Render the active examination interface.
    """
    q_num = st.session_state.question_number
    total_q = st.session_state.total_questions
    current_q = st.session_state.current_question

    st.subheader(f"Question {q_num} of {total_q}")
    st.progress(q_num / total_q)

    # Info bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Student", st.session_state.student_name)
    with col2:
        st.metric("Subject", st.session_state.subject)
    with col3:
        st.metric("Topic", current_q["topic"])
    with col4:
        st.metric("Difficulty", current_q["difficulty"])

    st.markdown("---")
    st.markdown(f"### **Q{q_num}: {current_q['question']}**")

    # Text area for student answer (clears or locks based on submission state)
    answer_text = st.text_area(
        "Your Answer",
        key=f"answer_text_area_{q_num}",
        height=180,
        disabled=st.session_state.answer_submitted,
        placeholder="Type your explanation here. Focus on core concepts and technical details...",
    )

    if not st.session_state.answer_submitted:
        if st.button("Submit Answer", type="primary"):
            submit_answer(answer_text)
            st.rerun()
    else:
        # Display evaluation feedback
        last_eval = st.session_state.last_evaluation
        if last_eval:
            st.markdown("---")
            st.markdown("#### Evaluation Feedback")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.metric("Score", f"{last_eval['score']:.1f} / 10")
            with f_col2:
                st.metric("Category", last_eval["category"])
            with f_col3:
                st.metric("Next Difficulty", last_eval["next_difficulty"])

        st.markdown("---")
        if q_num < total_q:
            if st.button("Next Question", type="primary"):
                load_next_question()
                st.rerun()
        else:
            st.success("Viva Completed! All questions have been evaluated.")
            if st.button("View Final Report", type="primary"):
                st.session_state.viva_completed = True
                st.rerun()


def show_final_report() -> None:
    """
    Render comprehensive final performance report and metrics.
    """
    st.title("🎓 VIVA COMPLETED")
    st.markdown("### Examination Performance Report")

    scores_data = st.session_state.scores
    if not scores_data:
        st.warning("No scores recorded.")
        return

    scores_list = [s["score"] for s in scores_data]
    avg_score = float(np.mean(scores_list))
    max_score = float(np.max(scores_list))
    min_score = float(np.min(scores_list))

    # Overall Performance category based on average score
    if avg_score >= 8.0:
        overall_perf = "Excellent"
    elif avg_score >= 6.0:
        overall_perf = "Good"
    elif avg_score >= 4.0:
        overall_perf = "Average"
    else:
        overall_perf = "Needs Improvement"

    # Summary Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Questions", len(scores_data))
    with c2:
        st.metric("Average Score", f"{avg_score:.2f} / 10")
    with c3:
        st.metric("Highest Score", f"{max_score:.2f}")
    with c4:
        st.metric("Lowest Score", f"{min_score:.2f}")

    st.markdown("---")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.write(f"**Student:** {st.session_state.student_name}")
        st.write(f"**Subject:** {st.session_state.subject}")
    with col_meta2:
        st.write(f"**Overall Performance:** `{overall_perf}`")
    with col_meta3:
        st.write(
            f"**Final Difficulty Level:** `{st.session_state.current_difficulty}`"
        )

    # Detailed Results Table
    st.markdown("### Question Breakdown")
    report_rows = []
    for s in scores_data:
        report_rows.append(
            {
                "Question": s["question"],
                "Topic": s["topic"],
                "Difficulty": s["difficulty"],
                "Score": f"{s['score']:.1f}",
                "Category": s["category"],
            }
        )
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

    # Topic Strengths and Weaknesses
    st.markdown("### Topic Mastery Analysis")
    topic_scores: Dict[str, List[float]] = {}
    for s in scores_data:
        t = s["topic"]
        topic_scores.setdefault(t, []).append(s["score"])

    strong_topics = []
    weak_topics = []
    for topic, t_scores in topic_scores.items():
        t_avg = np.mean(t_scores)
        if t_avg >= 7.0:
            strong_topics.append(f"{topic} (Avg: {t_avg:.1f})")
        elif t_avg < 5.0:
            weak_topics.append(f"{topic} (Avg: {t_avg:.1f})")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("**Strong Topics:**")
        if strong_topics:
            for st_item in strong_topics:
                st.write(f"✅ {st_item}")
        else:
            st.write("None identified (scores below 7.0 threshold).")

    with t_col2:
        st.markdown("**Weak Topics:**")
        if weak_topics:
            for wk_item in weak_topics:
                st.write(f"⚠️ {wk_item}")
        else:
            st.write("None identified (no topics below 5.0).")

    st.markdown("---")
    if st.button("Start New Viva", type="primary"):
        # Reset state back to initial screen
        st.session_state.viva_started = False
        st.session_state.viva_completed = False
        st.session_state.scores = []
        st.session_state.current_question = None
        st.session_state.answer_submitted = False
        st.session_state.last_evaluation = None
        st.session_state.question_engine.reset()
        st.session_state.adaptive_engine.reset("Easy")
        st.rerun()


def main():
    """
    Main Streamlit application entrypoint.
    """
    st.set_page_config(
        page_title="AI Viva Examiner",
        page_icon="🎓",
        layout="wide",
    )

    initialize_session_state()

    # 1. Sidebar Setup
    st.sidebar.title("AI Viva Examiner")
    st.sidebar.markdown("Adaptive Deep Learning Viva Examination")

    input_name = st.sidebar.text_input(
        "Student Name:",
        value=st.session_state.student_name,
        placeholder="e.g. John Doe",
    )

    input_subject = st.sidebar.text_input(
        "Subject:",
        value=st.session_state.subject,
        placeholder="e.g. Deep Learning",
    )

    topic_options = [
        "All Topics",
        "Neural Networks",
        "Activation Functions",
        "Backpropagation",
        "CNN",
        "RNN",
    ]
    input_topic = st.sidebar.selectbox("Topic:", topic_options)

    input_diff = st.sidebar.selectbox(
        "Initial Difficulty:", ["Easy", "Medium", "Hard"], index=0
    )

    input_num_q = st.sidebar.number_input(
        "Number of Questions:",
        min_value=3,
        max_value=10,
        value=st.session_state.total_questions,
        step=1,
    )

    if st.sidebar.button("Start Viva", type="primary"):
        start_viva(
            student_name=input_name,
            subject=input_subject,
            selected_topic=input_topic,
            initial_difficulty=input_diff,
            total_questions=input_num_q,
        )
        st.rerun()

    # 2. Main Content Routing
    if not st.session_state.viva_started:
        st.title("🎓 AI Viva Examiner")
        st.markdown(
            """
            Welcome to the **AI Viva Examiner** — an adaptive examination system powered by Deep Learning.

            ### Instructions:
            1. Enter your **Student Name** and **Subject** in the sidebar.
            2. Choose your preferred **Topic** or select **All Topics**.
            3. Set your starting difficulty and total questions (3 to 10).
            4. Click **Start Viva** to begin.
            5. Your answers will be analyzed by a trained **Bidirectional LSTM** model, and the difficulty of subsequent questions will automatically adapt based on your conceptual accuracy!
            """
        )
        st.info("👈 Please configure your examination settings in the sidebar to begin.")

    elif st.session_state.viva_completed:
        show_final_report()

    else:
        show_viva_screen()


if __name__ == "__main__":
    main()
