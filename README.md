# Spam-Mail-Detection-Using-Machine-Learning-And-Python
This project is a machine learning-based web application that detects whether a given email or SMS message is Spam or Not Spam.
It uses Natural Language Processing (NLP) techniques to preprocess text and a Multinomial Naive Bayes classifier to make predictions.
The app is built using Streamlit for an interactive and user-friendly interface.

# Technologies Used
    Python
    scikit-learn – for machine learning model training
    NLTK – for text preprocessing (tokenization, stopword removal, stemming)
    Streamlit – for building the web app interface
    Pandas, NumPy – for data handling and analysis
    Pickle – for saving and loading trained models

# Project Workflow
  ## 1. Data Collection
    The dataset used is the popular Spam SMS Dataset containing labeled messages (Spam/Ham).
  ## 2. Data Preprocessing
    Text messages are cleaned and transformed:
      Converted to lowercase
      Tokenized using NLTK
      Removed stopwords and punctuation
      Applied stemming using Porter Stemmer
  ## 3. Feature Extraction
    Used TF-IDF Vectorization to convert text into numerical feature vectors.
  ## 4. Model Training
    Trained a Multinomial Naive Bayes classifier on the processed dataset.
    Split the data into training and testing sets for performance evaluation.
  ## 5. Model Deployment
    The trained model and vectorizer were saved using pickle.
    A Streamlit app was created to allow users to input messages and get instant predictions.
# Installation & Setup
  ## 1. Create and activate a virtual environment (optional but recommended):
    python -m venv venv
    venv\Scripts\activate
  ## 2. Install dependencies:
    pip install -r requirement.txt
# Step-by-Step Usage
  ## A. Train the Model
    streamlit run train_model.py
  ## B. Run the app
    streamlit run app.py
