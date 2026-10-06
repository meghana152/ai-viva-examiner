# Deep Learning Model Documentation

This directory contains scripts and documentation for developing, training, and evaluating the Deep Learning model for the AI Viva Examiner.

## Architecture Plan
The evaluation engine will assess semantic similarity between a student's response and reference answers/key concepts. Planned Deep Learning architectures include:
- **Siamese Neural Network (LSTM / BiLSTM)**: Dual subnetworks processing student response and reference answer simultaneously to compute cosine similarity / Manhattan distance.
- **Feedforward Deep Neural Network**: Layered dense network taking sentence embeddings (e.g., pre-trained word vectors or TF-IDF + embedding features) and computing semantic correctness scores.

## Files in this Directory
- `train.py`: Script to build, compile, train, and save the Keras/TensorFlow model using `data/answer_dataset.csv`.
- `evaluate.py`: Script to load the trained model, compute test loss/metrics, generate classification/regression reports, and validate against validation datasets.
- Model artifact files (`.keras` or `.h5`) and tokenizers will be saved here upon model training.
