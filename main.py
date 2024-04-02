import pandas as pd
import numpy as np
import nltk
import pickle
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import *
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import streamlit as st
import boto3
from io import BytesIO
from st_files_connection import FilesConnection


conn = st.connection('s3', type=FilesConnection)
s3 = boto3.resource('s3')

with BytesIO() as data:
    s3.Bucket("productreviewsentiment").download_fileobj("sentiment.pkl", data)
    data.seek(0)    # move back to the beginning after writing
    model = pickle.load(data)

def process_comment(comment):
    # remove stock market tickers like $GE
    comment = re.sub(r'\$\w*', '', comment)
    # remove filler words like "\\n"
    comment = re.sub(r'[\n]', '', comment)
    # remove punctuation so that "!" does not get assumed by the model to be a positive review
    comment = re.sub(r'[?!]', '', comment)
    # remove old style text "RT"
    comment = re.sub(r'^RT[\s]+', '', comment)
    # remove hyperlinks
    comment = re.sub(r'https?:\/\/.*[\r\n]*', '', comment)
    # remove hashtags, only removing the hash # sign from the word
    comment = re.sub(r'#', '', comment)
    return comment

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def stop():
    stop = stopwords.words('english')
    return stop

def app_prediction(model, text):
    answer = int(model.predict(text)[0])
    if answer == 0:
        label = f"This review is negative :anguished:"
    if answer == 1:
        label = f"This review is positive :smiley:"
    return label

# App Interface
st.title("Product Review Positivity Predictor (positive or negative)")

# Input
text = st.text_input('Enter full review', 'Copy and paste the review here')
text = list(text)

# load model
with open("final.pkl", 'rb') as f:
    model = pickle.load(f)

# prediction
    # design user interface
if st.button("Find Out"):
    x = process_comment(text)
    prediction = app_prediction(model, x)
    st.subheader("Prediction based on your inputs:")
    st.write(prediction)
