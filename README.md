# AI Viva Examiner

An interactive, adaptive viva examination web application powered by Deep Learning that evaluates student textual responses and dynamically adjusts question difficulty in real time based on conceptual comprehension.

---

## Features

- **Adaptive viva questioning**: Dynamically selects questions matching the student's real-time skill level.
- **Deep Learning based answer evaluation**: Evaluates open-ended student responses using an embedded Bidirectional LSTM neural network.
- **BiLSTM answer scoring**: Predicts continuous conceptual quality scores from 0.0 to 10.0 and classifies responses into quality bands (Poor, Weak, Partial, Good, Excellent).
- **Difficulty adaptation**: Dynamically shifts difficulty between Easy, Medium, and Hard based on performance thresholds.
- **Topic-wise performance analysis**: Breaks down strengths and weaknesses across study topics.
- **Final viva report**: Summarizes total questions, score trajectory, category distribution, and topic mastery.

---

## Technology Stack

- Python
- TensorFlow/Keras
- BiLSTM
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## Project Structure

```text
.
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Project overview and setup documentation
├── .gitignore                 # Version control exclusions
│
├── data/
│   ├── questions.csv          # 50-question viva question bank across 5 topics
│   └── answer_dataset.csv     # 500-sample supervised answer dataset
│
├── model/
│   ├── viva_bilstm.keras      # Trained Bidirectional LSTM model weights
│   ├── tokenizer.json         # Fitted vocabulary and token mapping artifact
│   ├── train.py               # Model training pipeline
│   ├── evaluate.py            # Comprehensive evaluation and plotting script
│   ├── actual_vs_predicted.png# Model prediction scatter plot
│   └── residual_plot.png      # Error residual distribution plot
│
├── preprocessing/
│   ├── text_processor.py      # Keras Tokenizer and text cleaner
│   └── __init__.py
│
├── viva/
│   ├── question_engine.py     # Question loading, topic filtering, and state tracking
│   ├── adaptive_engine.py     # Adaptive difficulty state machine
│   ├── evaluator.py           # Answer evaluation and scoring interface
│   └── __init__.py
│
└── utils/
    └── helpers.py             # General helper routines
```

---

## How to Run Locally

1. **Clone the repository and navigate into the project directory**:
   ```bash
   git clone https://github.com/<your-username>/AI-Viva-Examiner.git
   cd AI-Viva-Examiner
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit web application**:
   ```bash
   streamlit run app.py
   ```

---

## Model

The answer evaluation engine is a **Bidirectional Long Short-Term Memory (BiLSTM)** regression network implemented in TensorFlow/Keras. It encodes preprocessed student response sequences through an embedding layer, bidirectional temporal recurrent layers, dropout regularization, and a dense regression head to predict an answer quality score from **0.0 to 10.0**.

> [!NOTE]
> The model acts as an automated semantic evaluation assistant estimating conceptual completeness and relevance against college viva reference rubrics; it does not claim to provide perfect factual correctness or replace formal faculty evaluation.

---

## Dataset

The question bank contains 50 verified technical viva questions covering 5 core Deep Learning subjects (Neural Networks, Activation Functions, Backpropagation, CNN, and RNN). 

The evaluation training dataset ([data/answer_dataset.csv](data/answer_dataset.csv)) is a **curated synthetic educational dataset** composed of 500 graded answers across distinct proficiency tiers (Poor, Weak, Partial, Good, and Excellent) developed specifically for training and prototyping the adaptive viva examination system.
