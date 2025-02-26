# News Article Classification
This project implements a machine learning pipeline for classifying news articles into different categories (World, Sports, Business, and Science/Technology) using Naive Bayes and Logistic Regression models. The pipeline includes text preprocessing, vectorization, model training, evaluation, and hyperparameter tuning.

## Steps

1. **Load and preprocess Data**
    - Load the dataset from Parquet files.
    - Clean the text by removing punctuation, digits, stopwords, and applying lowercasing.

2. **Text Vectorization**
    - Convert the text data into numerical features using the TF-IDF vectorizer.

3. **Model Training**
    - Train two classification models:
        - Multinomial Naive Bayes
        - Logistic Regression (with Grid Search for hyperparameter tuning)

4. **Model Evaluation**
    - Evaluate models using Precision, Recall, and F1-score metrics.

5. **Sample Predictions**
    - Predict categories for sample headlines.

## Requirements

- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn

## Usage
1. Install the required libraries:
```
pip install pandas nltk scikit-learn matplotlib seaborn
```
2. Download the dataset and place it in the appropriate folder.
3. Run the script to preprocess data, train models, and evaluate them.

## Example Predictions:

Input: "Stock market hits record high amid economic optimism"
Predicted Category: Business 📈

Input: "Champions League final ends in dramatic penalty shootout"
Predicted Category: Sports ⚽

Input: "NASA announces new mission to explore distant exoplanets"
Predicted Category: Science/Technology 🚀
