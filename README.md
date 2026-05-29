# 📰 Fake News Classifier Using LSTM

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)

An end-to-end Natural Language Processing (NLP) deep learning project that identifies unreliable and fake news articles. By leveraging a Long Short-Term Memory (LSTM) network and custom word embeddings, the model effectively captures the sequential context of news titles to classify them as either fake (1) or real (0).

## 🚀 Project Overview

With the rapid spread of misinformation, automated fake news detection is a critical application of AI. This project utilizes the [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data) to train a sequence model capable of understanding linguistic nuances and structural patterns in deceptive news titles.

### Key Highlights:
- **Architecture**: Embedding Layer → Dropout (0.3) → LSTM (100 units) → Dropout (0.3) → Dense (Sigmoid).
- **Text Preprocessing**: Tokenization, regex cleaning, Porter Stemming, and custom one-hot representation (Vocabulary size: 5,000).
- **Performance**: Achieved **90.37% Accuracy** with robust F1-scores across both classes.

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow / Keras
- **NLP**: NLTK (Stopwords, PorterStemmer)
- **Data Manipulation**: Pandas, NumPy
- **Evaluation**: Scikit-Learn (Confusion Matrix, Classification Report)
- **Environment**: Jupyter Notebook / VS Code

## 🧠 Methodology

1. **Data Cleaning**: Removed incomplete records (`dropna()`) to maintain sequence integrity.
2. **Text Processing**: 
   - Filtered non-alphabetic characters using Regex.
   - Converted text to lowercase and removed English stopwords.
   - Applied Porter Stemming to reduce words to their root forms.
3. **Vectorization**:
   - Converted text into one-hot encoded vectors based on a 5,000-word vocabulary.
   - Padded sequences (`maxlen=20`) to ensure uniform input dimensions for the LSTM.
4. **Model Training**:
   - Built a Sequential model with a 40-dimensional embedding space.
   - Integrated Dropout layers to prevent overfitting.
   - Trained over 10 epochs using the Adam optimizer and Binary Crossentropy loss.

## 📊 Results

The model was evaluated on a 33% holdout test set (6,035 samples), yielding the following metrics:

- **Overall Accuracy**: 90.37%
- **Precision**: 89% (Fake News), 91% (Real News)
- **Recall**: 89% (Fake News), 92% (Real News)
- **F1-Score**: 0.90 (Macro Avg)

## 💻 Installation & Usage

It is highly recommended to use `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone [https://github.com/PranavKumarG/fake-news-lstm.git](https://github.com/PranavKumarG/fake-news-lstm.git)
cd fake-news-lstm

# Initialize a virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv pip install pandas numpy tensorflow nltk scikit-learn jupyter

# Launch VS Code or Jupyter
jupyter notebook FakeNewsClassifierUsingLSTM.ipynb
