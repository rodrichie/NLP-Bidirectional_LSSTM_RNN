# NLP-Bidirectional_LSSTM_RNN
# Fake News Classifier using Bidirectional LSTM

## Overview
This notebook demonstrates the implementation of a Fake News Classifier using Bidirectional Long Short-Term Memory (LSTM) networks. The model is trained to differentiate between fake and real news articles based on the provided dataset.

## Dataset
- **Source**: [Kaggle Fake News Classification Competition](https://www.kaggle.com/competitions/fake-news/data?select=train.csv)
- **Columns**: 
  - `id`
  - `title`
  - `author`
  - `text`
  - `label`
- **Binary Classification**:
  - `1`: Fake News
  - `0`: Real News

## Key Steps in the Notebook
### Data Preprocessing
- Loading and exploring the dataset.
- Handling missing values.
- Tokenizing text data.
- One-hot encoding and padding sequences for embedding representation.

### Model Architecture
- **Embedding Layer**: Maps the input vocabulary to a dense representation.
- **Bidirectional LSTM**: Captures contextual dependencies in both forward and backward directions.
- **Dense Layer**: Outputs binary predictions using a sigmoid activation function.

### Model Training
- Splitting data into training and testing sets.
- Compiling the model with:
  - Loss: Binary Crossentropy
  - Optimizer: Adam
  - Metrics: Accuracy
- Training the model on the preprocessed data.

### Evaluation
- Predictions on the test set.
- Confusion matrix and accuracy score for model performance evaluation.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn
- NLTK

## How to Run the Notebook
1. Install the required libraries:
   ```bash
   pip install tensorflow keras pandas numpy scikit-learn nltk
   ```
2. Download the dataset from Kaggle.
3. Open the notebook and run the cells step by step.
4. Train the model using the dataset.
5. Evaluate the model's performance on the test set.

## Potential Improvements
- Experiment with varying sequence lengths and embedding dimensions.
- Use pre-trained word embeddings such as GloVe or Word2Vec.
- Fine-tune hyperparameters like the number of LSTM units and dropout rates.
- Incorporate additional layers for more complex architectures.

## License
This project is open-source and intended for educational purposes.

## Credits
- Dataset provided by [Kaggle](https://www.kaggle.com/competitions/fake-news/data?select=train.csv).
- Libraries used: TensorFlow, Keras, Pandas, NumPy, scikit-learn, NLTK.
