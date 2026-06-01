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
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
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
    packages = [
        "punkt", "punkt_tab", "stopwords",
        "wordnet", "omv-1.4", "averaged_percentron_tagger"
    ]
    print(" Checking NLTK data...")
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    print(" NLTK data ready. \n")

#--------------------------------------------------------------
# SECTION 2: Sample Dataset
# Concepts: Labeled text data - the foundation of 
#           supervized NLP, Each text has a sentiment label.
#--------------------------------------------------------------
REVIEWS = [
    # Positive reviews (label = 1)
    ("This product exceeded all my expectations, absolutely flawless performance.", 1),
    ("Incredibly fast shipping and the build quality is top-tier.", 1),
    ("Best purchase I have made all year, highly recommend it!", 1),
    ("The customer service team went above and beyond to help.", 1),
    ("Super intuitive interface, set it up in under five minutes.", 1),
    ("Beautifully packaged and works exactly as described in the ad.", 1),
    ("Five stars! I will definitely be buying from this brand again.", 1),
    ("The battery life is phenomenal, easily lasts me two full days.", 1),
    ("Exceeded my quality standards, feels very premium and durable.", 1),
    ("An absolute game-changer for my daily workflow, love it.", 1),
    ("So lightweight and portable, perfect for traveling long distances.", 1),
    ("The sound quality is crisp, deep bass and crystal clear vocals.", 1),
    ("Extremely comfortable to wear for hours at a time without pain.", 1),
    ("Amazing value for money, you won’t find a better deal.", 1),
    ("The software updates have made this device even better over time.", 1),
    ("Brilliant design, very modern aesthetic that fits perfectly in my room.", 1),
    ("Works flawlessly with all my other smart home devices instantly.", 1),
    ("A wonderfully engineered product that solves my exact issues perfectly.", 1),
    ("The instructions were crystal clear and assembly was a total breeze.", 1),
    ("Stays perfectly cool even during heavy, intense usage sessions.", 1),
    ("Highly impressive performance, runs smoother than my older expensive model.", 1),
    ("The materials used are premium, feels very sturdy in hand.", 1),
    ("A perfect gift choice, my friend was absolutely thrilled with it.", 1),
    ("Outstanding reliability, has not let me down a single time.", 1),
    ("Simple, elegant, and highly effective at what it is built for.", 1),

    # Negative reviews (label = 0)
    ("Complete waste of money, stopped working after just three days.", 0),
    ("The product looks absolutely nothing like the pictures online.", 0),
    ("Terrible customer support, they refused to issue a refund.", 0),
    ("Arrived completely broken and the packaging was completely shredded.", 0),
    ("The app constantly crashes and refuses to sync with my phone.", 0),
    ("Incredibly cheap plastic material, snapped during the first setup.", 0),
    ("The battery dies within thirty minutes of a full charge.", 0),
    ("Horribly loud and annoying buzzing noise while it is plugged in.", 0),
    ("Extremely disappointed with this purchase, do not buy it.", 0),
    ("The instructions are unreadable and parts were missing from the box.", 0),
    ("Completely useless features, does not do what it claims at all.", 0),
    ("The shipping took over a month and the item arrived damaged.", 0),
    ("Extremely uncomfortable to use, gave me a massive headache.", 0),
    ("Way overpriced for such low-grade quality and poor performance.", 0),
    ("The touch screen is completely unresponsive half of the time.", 0),
    ("It constantly overheats within minutes and shuts itself off completely.", 0),
    ("The software is full of bugs and glitches, completely unusable.", 0),
    ("Falsely advertised specs, it is much smaller than the description states.", 0),
    ("The color faded drastically after just one gentle wash cycle.", 0),
    ("Customer service ignored my emails for two weeks straight.", 0),
    ("A total nightmare to assemble, holes do not line up at all.", 0),
    ("The connection keeps dropping every single time I try to use it.", 0),
    ("Feels like a cheap knockoff brand, definitely returning this immediately.", 0),
    ("It emitted a strange burning smell the first time I turned it on.", 0),
    ("Save your money and look for a completely different alternative.", 0),

    # Neutral/Mixed    (label = 2)
    ("The screen looks gorgeous, but the battery life is quite lacking.", 2),
    ("It gets the job done fine, though it feels a bit overpriced.", 2),
    ("The shipping was incredibly fast, but the item had minor scratches.", 2),
    ("Decent build quality, but the setup process was unnecessarily complicated.", 2),
    ("The hardware is excellent, but the companion software needs heavy updates.", 2),
    ("Works perfectly fine, nothing special or groundbreaking about it though.", 2),
    ("An okay product, has some great pros but equally annoying cons.", 2),
    ("The sound is great at low volumes, but distorts heavily when loud.", 2),
    ("It fits well and looks nice, but the material is a bit scratchy.", 2),
    ("Customer service was helpful, but the replacement item took forever to arrive.", 2)
]

def get_dataframe():
    """ Convert review list to a labeled DataFrame."""
    df        = pd.DataFrame(REVIEWS, columns=["text", "sentiment"])
    label_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
    df["label"] = df["sentiment"].map(label_map)
    return df

#--------------------------------------------------------------
# SECTION 3: Text Preprocessing
# Concepts: Raw text is messy - clean it before 
#           feeding to any NLP model
#--------------------------------------------------------------

stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()

def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation & special characters
    3. Tokenize (split into words)
    4. Remove stopwords (the, is a, an...)
    5. Stem or Lemmatize
    """
    # Step 1: Lowercase - "Amazing" and "amazing" are the same word
    text = text.lower()

    # Setp 2: Remove URLs, emails, numbers, punctuation
    text = re.sub(r"http\S+|wwwS+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "", text).strip()

    # Step 3: Tokenize -- split string into list of words
    tokens = word_tokenize(text)

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Step 5: Stem OR Lemmatize (not both)
    if use_stemming:
        # Stemming: chops suffix -- "running" -> "run", "happily" -> "happili" (crude)
        tokens = [stemmer.stem(t) for t in tokens]
    elif use_lemmatization:
        # Lemmatization: uses dictionary -- "running" -> "run", "better" -> "good"
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def show_preprocessing_demo(df):
    """ Show before/after preprocessing for 3 examples. """
    print("\n ===== Text Preprocessing Demo ============")
    for _, row in df.head(3).iterrows():
        cleaned = preprocess_text(row["text"])
        print(f"\n Original    : {row["text"]}")
        print(f" Cleaned    : {cleaned}")
        print(f" Label      : {row["label"]}")


#--------------------------------------------------------------
# SECTION 4: Tokenization Deep Dive
# Concepts: word_tokenize, sent_tokenize,
#           FreqDist, bigrams/trigrams (n-grams)
#--------------------------------------------------------------

def tokenization_demo(df):
    """ Demonstrate tokenization, frequency distribution, and n-grams."""
    print("\n ====== Tokenization & N-Grams =======================")

    # Combine all positive reviews into one corpus
    pos_text = " ".join(df[df["label"] == "Positive"]["text"].tolist())
    neg_text = " ".join(df[df["label"] == "Negative"]["text"].tolist())

    # Sentence tokenization
    sentences = sent_tokenize(pos_text)
    print(f"\n Positive corpus: {len(sentences)} sentences")

    # Word tokenization + frequency distributio 
    pos_tokens = [w.lower() for w in word_tokenize(pos_text)
                  if w.isalpha() and w.lower() not in stopwords.words("english")]
    neg_tokens = [w.lower() for w in word_tokenize(neg_text)
                  if w.isalpha() and w.lower() not in stopwords.words("english")]
    
    pos_freq    = FreqDist(pos_tokens)
    neg_tokens  = FreqDist(neg_tokens)

    print(f"\n Top 10 Positive words.")
    for word, count in pos_freq.most_common(10):
        bar = " " * count
        print(f" {word:<15} {count:>3} {bar}")
    
    # Bigrams -- pairs of consecutive words
    # "great quality" appears together -- more iformative than "great" alone
    pos_bigrams = list(bigrams(pos_tokens))
    pos_bigrams_freq = FreqDist(pos_bigrams)
    print(f"\n Top 5 Positive Biagrams (word pairs):")
    for bg, count in pos_bigrams_freq.most_common(5):
        print(f" {bg[0]} + {bg[1]:<15} {count}")


#--------------------------------------------------------------
# SECTION 5: TF-IDF Vectorization
# Concepts: Convert text --> numbers that model can use
#
# TF = Term Frequency -- how often word appears in THIS document
# IDF = Inverse Doc Freq -- penalizes words common in ALL documents
# TF-IDF = TF x IDF -- high score = word is important HERE but rare elsewhere
#--------------------------------------------------------------

def show_tfidf_demo(df):
    """Show what TF-IDF actually produces."""
    print("\n ======= TF-IDF Vectorization Demo ======================")

    sample_texts = df["text"].head(5).tolist()

    #CountVectorizer -- simple word counts (bag of words)
    count_vec       = CountVectorizer(max_features=10, stop_words="english")
    count_matrix    = count_vec.fit_transform(sample_texts)
    print(f"\n CountVectorizer vocabulary (top 10):")
    print(f" {list(count_vec.vocabulary_.keys())}")
    print(f" Matrix shape: {count_matrix.shape} (5 docs x 10 features)")

    # TF-IDF -- smarter weighting
    tfidf_vec       = TfidfVectorizer(max_features=10, stop_words="english")
    tfidf_matrix    = tfidf_vec.fit_transform(sample_texts)
    print(f"\n TF-IDF feature names: {tfidf_vec.get_feature_names_out()}")

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarrary(),
        columns=tfidf_vec.get_feature_names_out()
    ).round(3)
    print(f"\n TF-IDF scores (5 docs * 10 words):")
    print(tfidf_df.to_string())

#--------------------------------------------------------------
# SECTION 6: Build & Compare NLP Models
# Concepts: 3 clasifiers commonly used for text:
#           Naive Bayes - probabilistic, fast, great baseline
#           Logistic Regression - Linear, very strong for text
#           LinearSVC - SVM for text, often best performer
#--------------------------------------------------------------

def train_models(df):
    """ Train and evaluate 3 NLP classifiers."""
    print("\n ====== Training NLP Models ===========================")

    # Train/test split - stratify keeps class ratios equal in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),     # unigrams + bigrams
                max_features=5000,
                sublinear_tf=True       # log scaling of TF
            )),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),    
                max_features=5000,
                sublinear_tf=True       
            )),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            ))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),    
                max_features=5000,
                sublinear_tf=True       
            )),
            ("clf", LinearSVC(
                C=1.0, max_iter=2000, random_state=42
            ))
        ]),
    }

    results = []
    trained_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred      = pipeline.predict(X_test)
        acc         = accuracy_score(y_test, y_pred)
        f1          = f1_score(y_test, y_pred, average="weighted")
        cv_scores   = cross_val_score(pipeline, X, y, cv=5,
                                      scoring="accuracy", n_jobs=-1)
    
        print(f" {name:<25} {acc:>10.4f} {f1:>10.4f} {cv_scores.mean():>10.4f}")

        results.append({
            "model": name, "accuracy": acc,
            "f1": f1, "cv_accuracy": cv_scores.mean()
        })
        trained_models[name] = pipeline

    return pd.DataFrame(results), trained_models, X_test, y_test


#--------------------------------------------------------------
# SECTION 7: Detailed Evaluation
# Concepts: Confusion matrix, classification report
#           Precision, Recall, F1 - go beyond simple accuracy
#--------------------------------------------------------------

def detailed_evaluation(model, X_test, y_test, model_name="Best Model"):
    """Show confusion matrix and full classification report."""
    y_pred = model.predict(X_test)

    print(f"\n ====== Detailed Evaluation: {model_name} ==============")

    # Classification report -per-class precision, recall, F1
    # Precision = of all predicted Positive, how many were actually Positive?
    # Recall    = of all actual Positive, how many did we catch?
    # F1        = harmonic mean of Precision and Recall
    target_names = ["Negative", "Positive", "Neutral"]
    # Only use labels present in test set
    present_labels = sorted(set(y_test))
    present_names  = [target_names[l] for l in present_labels]

    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred,
                                labels=present_labels,
                                target_names=present_names))
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    fig, ax = plt.subplot(figsize=(7,5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a2e")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        linewidths=1, linecolor="#0f0f0f",
        ax=ax, char=False,
        annot_kws={"size":14, "color": "white"}
    )
    ax.set_title(f"Confusion Matrix - {model_name}",
                 color="white", fontsize=12)
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("Actual", color="#aaa")
    ax.tick_params(colors="#aaa")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=130,
                bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: confusion_matrix.png")
    plt.show(); plt.close()

#--------------------------------------------------------------
# SECTION 8: Word Frequency Visualization
# Concepts: Bar charts of most important words
#           per sentiment class
#--------------------------------------------------------------

def plot_word_frequencies(df):
    """ Plot top words per sentiment class."""
    df["clean_text"] = df["text"].apply(preprocess_text)

    fig, axes = plt.subplot(1, 3, figsize=(18, 6))
    fig.suptitle("Top Words by Sentiment", fontsize=14, color="white")
    fig.patch.set_facecolor("#0f0f0f")

    sentiment_colors = {
        "Positive": "#05c46b",
        "Negative": "#e94560",
        "Neutral" : "#ffd460"
    }

    for ax, (label, color) in zip(axes, sentiment_colors.items()):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")

        subset_texts = df[df["label"] == label]["clean_text"]
        all_words    = " ".join(subset_texts).split()
        freq         = FreqDist(all_words)
        top_words    = freq.most_common(12)

        if top_words:
            words, counts = zip(*top_words)
            bars = ax.barh(list(words)[::-1], list(counts)[::-1],
                           color=color, edgecolor="#0f0f0f", alpha=0.05)
            ax.set_title(f"{label} Reviews")
            ax.set_xlabel("Frequency")
            ax.grid(axis="x", alpha=0.3, color="#444")
    
    plt.tight_layout()
    plt.savefig("word_frequencies.png", dpi=130,
                bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: word_frequencies.png")
    plt.show(); plt.close()

#--------------------------------------------------------------
# SECTION 9: TF-IDF Top Features per Class
# Concepts: Which words most strongly signal each sentiment?
#--------------------------------------------------------------

def plot_tfidf_ton_feature(df):
    """ Show top TF-IDF features per class."""
    df["clear_text"] = df["text"].apply(preprocess_text)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000,
                            sublinear_tf=True)
    X = tfidf.fit_transform(df["clean_text"])
    feature_names = np.array(tfidf.get_feature_names_out())

    fig, axes = plt.subplot(1, 3, figsize=(18, 6))
    fig.suptitle("Top TF-IDF Features by Sentiment", fontsize=14, color="white")
    fig.patch.set_facecolor("#0f0f0f")

    labels = [0, 1, 2]
    names  = ["Negative", "Positive", "Neutral"]
    colors = ["#e94560", "#05c46b", "#ffd460"]

    for ax, label, name, color in zip(axes, labels, names, colors):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaa")
        ax.title.set_color("white")

        # Mean TF-IDF per feature for this class
        class_mask = (df["sentiment"] == label).values
        if class_mask.sum() == 0:
            continue

        class_tfidf  = X[class_mask].toarray().mean(axis=0)
        top_indices  = class_tfidf.argsort()[-12:]
        top_features = feature_names[top_indices]
        top_scores   = class_tfidf[top_indices]

        ax.barh(top_features, top_scores, color=color,
                 edgecolor="#0f0f0f", alpha=0.85)
        ax.set_title(f"{name} - Top TF-IDF Features")
        ax.set_xlabel("Mean TF-IDF Score")
        ax.grid(axis="x", alpha=0.3, color="#444")
    
    plt.tight_layout()
    plt.savefig("tfidf_features.png", dpi=130,
                bbox_inches="tight", facecolor="#0f0f0f")
    print(" Saved: tfidf_features.png")
    plt.show(); plt.close()

#--------------------------------------------------------------
# SECTION 10: Real-time Sentiment Predictor
# Concepts: Using the trained pipeline on brand new text
#--------------------------------------------------------------

label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
confidence_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

def predict_sentiment(model, text):
    """Predict sentiment of any input text."""
    cleaned    = preprocess_text(text)
    prediction = model.predict([cleaned])[0]
    label      = label_map.get(prediction, "Unknown")

    # Confidence via decision function or predict_proba
    try:
        proba       = model.predict_proba([cleaned])[0]
        confidence  = max(proba) * 100
        conf_label  = confidence_labels[min(3, int(confidence // 25))]
    except AttributeError:
        # LinearSVC doesn't have predict_proba - use decision function
        decision    = model.decision_function([cleaned])[0]
        if hasattr(decision, "__len__"):
            confidence = (max(decision) / (sum(abs(d) for d in decision) + 1e-9)) * 100
        else:
            confidence = abs(decision) / (abs(decision) + 1) * 100
        conf_label = confidence_labels[min(3, int(confidence // 25))]
    
    print(f"""
------------------------------------------------------------------
| Input         : {text[:45]:<45}|
| Cleaned       : {cleaned[:45]:<45}|
| Sentiment     : {label:<42}|
| Confidence    : {confidence:.1f}% ({conf_label}){' '* 20}|
------------------------------------------------------------------""")
    
    return prediction, confidence
def interactive_predictor(model):
    """Let user type any text and see prediction."""
    print("\n Real-time Sentiment Predictor")
    print(" Type any text (or 'back' to return)\n")

    test_sentances =[
        "This is absolutely wonderful. I love it!",
        "Terrible experience, never buying this again.",
        "It's okay I guess, nothing special.",
        "Best product I have ever used in my entire life",
        "Completely broken on arrival, very disasppointed.",
    ]

    print(" Demo predictions on sample sentences:")
    for sentence in test_sentances:
        predict_sentiment(model, sentence)
    
    print("\n  Now try your own: \n")
    while True:
        text = input(" Enter text: ").strip()
        if text.lower() in ("back", "exit", "0"):
            break
        if text:
            predict_sentiment(model, text)

#--------------------------------------------------------------
# SECTION 11: Full NLP Pipeline
#--------------------------------------------------------------

def run_full_pipeline():
    """Execute the complete NLP workflow. """
    print("\n Running Full NLP Pipeline... \n")

    download_nltk_data()

    # Step 1: Data
    print(" [1/6] Loading dataset...")
    df = get_dataframe()
    print(f" Shape: {df.shape}")
    print(f" {df['label'].value_counts().to_dict()}")

    # Step 2: Preprocessing demo
    print(" [2/6] Text Preprocessing...")
    show_preprocessing_demo(df)

    # Step 3: Tokenization
    print(" [3/6] Tokenization & N-Grams...")
    tokenization_demo(df)

    # Step 4: TF-IDF demo
    print(" [4/6] TF-IDF Vectorization...")
    show_tfidf_demo(df)

    # Step 5: Train Models
    print(" [5/6] Traning Models ...")
    results_df, trained_models, X_test, y_test = train_models(df)

    # Step 6: Best Model Evaluation
    best_name  = results_df.sort_values("f1", ascending=False).iloc[0]["model"]
    best_model = train_models[best_name]
    print(f"\n Best Model: {best_model}")

    print(" [6/6] Evaluation & Visualizations ...")
    detailed_evaluation(best_model, X_test, y_test, best_name)
    plot_word_frequencies(df)
    plot_tfidf_ton_feature(df)

    # Save Model
    joblib.dump(best_model, "sentiment_model.pkl")
    print(" Model saved: sentiment_model.pkl")

    print("\n Full NLP pipeline completed!")
    return best_model


#--------------------------------------------------------------
# SECTION 12 : Main Menu
#--------------------------------------------------------------
active_model = None

def main():
    global active_model

    print("\n" + "="* 52)
    print(" NLP SENTIMENT ANALYZER ")
    print("="*52)

    download_nltk_data()

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Run Full NLP Pipeline (start here)
              [2]  Preprocessing Demo
              [3]  Tokenization & N-Grams
              [4]  TF-IDF Vectorization Demo
              [5]  Train & Compare Models
              [6]  Plot TF-IDF Top Features
              [7]  Plot TF-IDF Top Features
              [8]  Load Saved Model
              [9]  Real-time Sentiment Predictor
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()
        df = get_dataframe()

        try:
            if choice == "1":
                active_model, *_ = run_full_pipeline()

            elif choice == "2":
                show_preprocessing_demo(df)

            elif choice == "3":
                tokenization_demo(df)

            elif choice == "4":
                show_tfidf_demo(df)

            elif choice == "5":
                results, models, X_test, y_test = train_models(df)
                best_name = results.sort_values("f1", ascending=False).iloc[0]["model"]
                active_model = models[best_name]
                detailed_evaluation(active_model, X_test, y_test, best_name)
            
            elif choice == "6":
                plot_word_frequencies(df)
            
            elif choice == "7":
                plot_tfidf_ton_feature(df)
            
            elif choice == "8":
                if os.path.exists("sentiment_model.pkl"):
                    active_model = joblib.load("sentiment_model.pkl")
                    print(" Model loaded.")
                else:
                    print(" No saved model found. Run option [1] first.")
            
            elif choice == "9":
                if active_model is None:
                    print(" Train a model first (option 1 0r 5)")
                else:
                    interactive_predictor(active_model)

            elif choice == "0":
                print("\n GoodBye!!!\m")
                break

            else:
                print(" Invalid Choice, try again!")
        
        except Exception as e:
            print(f" Error: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()