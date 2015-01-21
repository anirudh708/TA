'''
Created on Jan 19, 2015

@author: Gramener-pc

'''
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

model = Word2Vec.load_word2vec_format('300features_40minwords_10context', binary=True)