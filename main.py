import pandas as pd
import nltk
from flask import request
from flask import jsonify
from flask import Flask, render_template
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
import string
import pandas as pd
import numpy as np
import emoji


app = Flask(__name__)

@app.route('/')
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
    
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    text = list(text)
    with open("logistic_regression.pkl", 'rb') as f:
        model = pickle.load(f)
    answer = int(model.predict(text)[0])
    neg_face = emoji.emojize(":grimacing_face:")
    pos_face = emoji.emojize(":grinning_face:")
    if answer == 0:
        label = f"This review is negative {neg_face}"
    if answer == 1:
        label = f"This review is postivie! {pos_face}"

    return(render_template('index.html', variable=label))


if __name__ == "__main__":
    app.run(port='8088', threaded=False, debug=True)