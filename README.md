# Chatbot_using_RNN_with_Embedding_Transfer_Learning-
Neural Intent Classification and Response System
An end-to-end Natural Language Processing (NLP) pipeline designed to classify user intents and generate contextual responses using Recurrent Neural Networks (RNNs). This project demonstrates the implementation of deep learning architectures to solve sequence-to-sequence mapping problems.

Technical Highlights
1. Data Engineering & NLP Pipeline
Preprocessing: Developed a custom pipeline for text normalization, including noise reduction (punctuation removal) and case folding.

Tokenization: Leveraged Keras Tokenizer to vectorize natural language into indexed sequences for neural input.

Padding: Implemented post-sequence padding to ensure consistent input dimensionality across the entire dataset.

Label Encoding: Integrated Scikit-Learn LabelEncoder to transform multi-class categorical intent tags into machine-readable formats.

2. Model Architecture & Optimization
Built using a Sequential API in TensorFlow, the architecture includes:

Embedding Layer: Learns dense vector representations of the vocabulary in a 10-dimensional space.

Gated Recurrent Units (GRU) & LSTM: Utilized recurrent layers to capture long-range temporal dependencies and context within user queries.

Overfitting Control: Integrated Dropout layers and EarlyStopping callbacks. The system monitors loss during 2000 epochs and automatically restores the best-performing weights to ensure maximum generalization.

Inference Engine: Designed a real-time testing function that processes raw string inputs and maps model predictions back to human-readable responses.

Repository Structure
main.py: The core script for data loading, training, and live inference.

Intent.json: The underlying dataset containing intent patterns and response mappings.

requirements.txt: Environment configuration for project replication.

Installation and Usage
Clone the repository:

Install dependencies:

Bash
pip install -r requirements.txt
Execute the pipeline:

Bash
python main.py
Performance Evaluation
The system includes automated visualization tools using Matplotlib to monitor training accuracy and loss curves. A final evaluation phase runs before the live testing loop to verify model reliability on the training distribution.

License
This project is licensed under the MIT License.
