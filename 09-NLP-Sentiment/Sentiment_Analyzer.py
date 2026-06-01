"""
Project 9: NLP Sentiment Analyzer
Concepts: text processing, tokenization, stopwords,
          stemming/lemmatization, TF-IDF vectorization,
          Naive Bayes, Logistic Regression, SVM Classifiers,
          confusion matrix, classification report,
          word clouds, n-grams, custom text prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
import os
warnings.filterwarnings("ignore")

# ------ LNP Libraries ------------------------------------------
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import bigrams, trigrams

# ------ scikit-learn ------------------------------------------
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import joblib

#--------------------------------------------------------------
# SECTION 1: Download NLTK Data
# Concepts: NLTK requires downloading language data
#           (dictionaries, tokenizers) separately
#--------------------------------------------------------------

def download_nltk_data():
    """ Download required NLTK datasets if not already present."""
    
#--------------------------------------------------------------
# SECTION 2: Sample Dataset
# Concepts: Labeled text data - the foundation of 
#           supervized NLP, Each text has a sentiment label.
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 3: Text Preprocessing
# Concepts: Raw text is messy - clean it before 
#           feeding to any NLP model
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 4: Tokenization Deep Dive
# Concepts: word_tokenize, sent_tokenize,
#           FreqDist, bigrams/trigrams (n-grams)
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 5: TF-IDF Vectorization
# Concepts: Convert text --> numbers that model can use
#
# TF = Term Frequency -- how often word appears in THIS document
# IDF = Inverse Doc Freq -- penalizes words common in ALL documents
# TF-IDF = TF x IDF -- high score = word is important HERE but rare elsewhere
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 6: Build & Compare NLP Models
# Concepts: 3 clasifiers commonly used for text:
#           Naive Bayes - probabilistic, fast, great baseline
#           Logistic Regression - Linear, very strong for text
#           LinearSVC - SVM for text, often best performer
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 7: Detailed Evaluation
# Concepts: Confusion matrix, classification report
#           Precision, Recall, F1 - go beyond simple accuracy
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 8: Word Frequency Visualization
# Concepts: Bar charts of most important words
#           per sentiment class
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 9: TF-IDF Top Features per Class
# Concepts: Which words most strongly signal each sentiment?
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 10: Real-time Sentiment Predictor
# Concepts: Using the trained pipeline on brand new text
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 11: Full NLP Pipeline
#--------------------------------------------------------------

#--------------------------------------------------------------
# SECTION 12 : Main Menu
#--------------------------------------------------------------
active_model = None