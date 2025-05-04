# Automatic Personality Prediction

This project aims to predict personality types using textual data from the Myers-Briggs Personality Type Indicator (MBTI) dataset. The goal is to leverage machine learning and natural language processing (NLP) techniques to classify users based on their writing style.

## Project Overview

The project explores both **supervised** and **unsupervised** machine learning methods for predicting MBTI personality types:

- **Supervised Methods**: We implemented Logistic Regression, Multinomial Naive Bayes, and Long Short-Term Memory (LSTM) networks to predict personality types based on text data.
- **Unsupervised Methods**: An Autoencoder combined with KMeans clustering was used to discover latent personality patterns from the text, without relying on labels.

## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/datasnaek/mbti-type/data). It contains over 8,600 rows of data, with each row including a person's self-reported MBTI personality type and a collection of their last 50 online posts in text format.

## Key Steps

1. **Data Preprocessing**: 
   - Text data was cleaned, tokenized, and lemmatized.
   - TF-IDF (Term Frequency-Inverse Document Frequency) was used for feature extraction.
   - SMOTE was applied to handle class imbalance.

2. **Model Training**: 
   - Various machine learning models were trained using the preprocessed data.
   - An LSTM model with GloVe embeddings and attention mechanisms was implemented to capture long-term dependencies in text.

3. **Model Evaluation**:
   - The models were evaluated using metrics like Accuracy, F1-score, and AUC-ROC.
   - Logistic Regression performed the best among the supervised models, with an AUC of 0.92.
   - The unsupervised approach revealed meaningful clusters based on writing style.

## Results

- Logistic Regression: Achieved the highest accuracy and AUC, outperforming other models.
- Custom LSTM: Showed promise with an AUC of 0.70 but struggled with the complexity of classifying all MBTI types.
- Unsupervised Clustering: The autoencoder and KMeans clustering produced well-separated clusters, indicating that writing style reflects latent personality traits.
