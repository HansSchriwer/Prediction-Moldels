#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Advanced Feature Engineering for API stock fetching


# In[ ]:


get_ipython().system(' pip install requests pandas scikit-learn numpy lightgbm catboost transformers')


# In[ ]:


import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import re
import io
import logging

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from transformers import BertTokenizer, BertModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
import shap

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SEC_EDGAR_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=S-1&count=100&output=atom"
USER_AGENT = "YourName your.email@example.com"  # Replace with your information
HEADERS = {'User-Agent': USER_AGENT}

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def fetch_recent_s1_filings():
    """
    Fetches recent S-1 filings using SEC EDGAR RSS feed.
    """
    response = requests.get(SEC_EDGAR_RSS_URL, headers=HEADERS)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    
    namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', namespaces)
    
    filings = []
    for entry in entries:
        company_name = entry.find('atom:title', namespaces).text
        filing_date_str = entry.find('atom:updated', namespaces).text
        filing_date = datetime.strptime(filing_date_str, '%Y-%m-%dT%H:%M:%S-04:00').date()
        link = entry.find('atom:link', namespaces).attrib['href']
        
        filings.append({
            'companyName': company_name,
            'filingDate': filing_date,
            'filingURL': link
        })
    logging.info(f"Fetched {len(filings)} recent S-1 filings.")
    return pd.DataFrame(filings)

def download_filing_document(filing_url):
    """
    Downloads the primary document from the filing page.
    """
    response = requests.get(filing_url, headers=HEADERS)
    response.raise_for_status()
    
    # Find the document link
    match = re.search(r'href="(.*?)"', response.text)
    if match:
        document_link = 'https://www.sec.gov' + match.group(1)
        doc_response = requests.get(document_link, headers=HEADERS)
        doc_response.raise_for_status()
        return doc_response.text
    return ""

def preprocess_text(text):
    """
    Preprocesses text for sentiment analysis and TF-IDF.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english') and len(word) > 2]
    return ' '.join(tokens)

def get_bert_embeddings(text):
    """
    Generates BERT embeddings for the given text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding.flatten()

def perform_topic_modeling(processed_texts, num_topics=10):
    """
    Performs LDA topic modeling and returns topic distribution features.
    """
    tokenized_texts = [text.split() for text in processed_texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    topic_features = [[topic_prob for _, topic_prob in doc] for doc in topics]
    topic_df = pd.DataFrame(topic_features, columns=[f'topic_{i}' for i in range(num_topics)])
    return topic_df

def get_vader_sentiment(text):
    """
    Computes VADER sentiment score for the given text.
    """
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def classify_industry(company_name):
    """
    Simple industry classification based on company name keywords.
    """
    tech_keywords = ['tech', 'software', 'internet', 'solutions', 'systems']
    finance_keywords = ['finance', 'financial', 'capital', 'investment']
    healthcare_keywords = ['health', 'medical', 'biotech', 'pharma']
    
    name_lower = company_name.lower()
    if any(word in name_lower for word in tech_keywords):
        return 'Technology'
    elif any(word in name_lower for word in finance_keywords):
        return 'Finance'
    elif any(word in name_lower for word in healthcare_keywords):
        return 'Healthcare'
    else:
        return 'Other'

def extract_features(df):
    """
    Extracts advanced features from the filings.
    """
    sentiments = []
    bert_embeddings = []
    processed_texts = []
    
    for idx, row in df.iterrows():
        logging.info(f"Processing {row['companyName']}...")
        try:
            doc_text = download_filing_document(row['filingURL'])
            preprocessed_text = preprocess_text(doc_text)
            df.at[idx, 'processedText'] = preprocessed_text
            sentiments.append(get_vader_sentiment(preprocessed_text))
            bert_emb = get_bert_embeddings(preprocessed_text)
            bert_embeddings.append(bert_emb)
        except Exception as e:
            logging.error(f"Error processing {row['companyName']}: {e}")
            df.at[idx, 'processedText'] = ""
            sentiments.append(0.0)
            bert_embeddings.append(np.zeros(768))  # BERT base hidden size
        time.sleep(0.5)  # Respect SEC's rate limits
    
    df['vader_sentiment'] = sentiments
    
    # Convert BERT embeddings to DataFrame
    bert_df = pd.DataFrame(bert_embeddings, columns=[f'bert_{i}' for i in range(768)])
    df = pd.concat([df, bert_df], axis=1)
    
    # Topic Modeling
    topic_df = perform_topic_modeling(df['processedText'].tolist(), num_topics=10)
    df = pd.concat([df, topic_df], axis=1)
    
    # Additional Features
    df['days_since_filing'] = (datetime.now().date() - df['filingDate']).dt.days
    df['is_tech'] = df['companyName'].str.contains('Tech|Solutions|Systems|Software|Internet', case=False).astype(int)
    df['industry'] = df['companyName'].apply(lambda x: classify_industry(x))
    le = LabelEncoder()
    df['industry_encoded'] = le.fit_transform(df['industry'])
    
    return df

def generate_target_variable(df):
    """
    Generates a target variable for demonstration purposes.
    In reality, this should be based on actual financial performance post-filing.
    """
    # Placeholder: Randomly assign high_return
    np.random.seed(42)
    df['high_return'] = np.random.randint(0, 2, size=len(df))
    return df

def build_model(df):
    """
    Builds and evaluates a sophisticated machine learning model.
    """
    # Define feature columns
    bert_features = [f'bert_{i}' for i in range(768)]
    topic_features = [f'topic_{i}' for i in range(10)]
    feature_cols = ['days_since_filing', 'is_tech', 'vader_sentiment', 'industry_encoded'] + bert_features + topic_features
    
    X = df[feature_cols]
    y = df['high_return']
    
    # Handle missing values if any
    X.fillna(0, inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define base learners
    base_learners = [
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression()
    
    # Stacking Classifier
    stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    
    # Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'final_estimator__C': [0.1, 1.0, 10.0]
    }
    
    grid_search = GridSearchCV(stacking_clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1]
    
    logging.info("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # SHAP Explanation
    explainer = shap.Explainer(best_model.named_steps['xgb'])
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    # Assign probability scores to the entire dataset
    df['return_probability'] = best_model.predict_proba(X)[::,1]
    
    return df.sort_values(by='return_probability', ascending=False), best_model

def main():
    # Step 1: Fetch recent S-1 filings
    logging.info("Fetching recent S-1 filings...")
    new_filings = fetch_recent_s1_filings()
    if new_filings.empty:
        logging.warning("No new S-1 filings found.")
        return
    
    # Step 2: Extract advanced features
    logging.info("Extracting features...")
    new_filings = extract_features(new_filings)
    
    # Step 3: Generate target variable (Placeholder)
    new_filings = generate_target_variable(new_filings)
    
    # Step 4: Build and evaluate the model
    logging.info("Building and evaluating the model...")
    analyzed_filings, model = build_model(new_filings)
    
    # Step 5: Display top potential companies with high probability
    threshold = 0.8  # Example threshold
    top_potential = analyzed_filings[analyzed_filings['return_probability'] >= threshold].head(10)
    logging.info("\nTop 10 Companies with High Return Potential:")
    print(top_potential[['companyName', 'filingDate', 'return_probability']])
    
    # Optionally, save the results
    analyzed_filings.to_csv('enhanced_analyzed_filings.csv', index=False)
    logging.info("\nResults saved to 'enhanced_analyzed_filings.csv'.")

if __name__ == "__main__":
    main()


# In[ ]:




