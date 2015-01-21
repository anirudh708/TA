''' 
code for sentiment analysis

'''

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec,Doc2Vec
from sklearn.ensemble import RandomForestClassifier

from s import KaggleWord2VecUtility
from sklearn import metrics
import gensim
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy
import math
import sys
from numpy import random,uint32
import traceback

# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews


def MyLabeledLineSentence(train_sen_len, model,  reviews):
    sens = []
    inner_id = 0
    try:
        for line in reviews:
            
            item_no =  train_sen_len + inner_id
            label = 'SENT_'+str(item_no)
            newvocab = gensim.models.doc2vec.Vocab()
            newvocab.index = item_no
            newvocab.sample_probability = 1.0
            newvocab.code = []
            for i in range(0, int(math.log(item_no, 2)+1)):
                newvocab.code.append(1)
            model.vocab[label] = newvocab
            model.syn0 = numpy.vstack((model.syn0, model.syn0[0]))
            model.index2word.append(label)
            random.seed(uint32(model.hashfxn(model.index2word[item_no] + str(model.seed))))
            model.syn0[item_no] = (random.rand(model.layer1_size) - 0.5) / model.layer1_size
            sens.append(gensim.models.doc2vec.LabeledSentence(line, ['SENT_%s' % item_no]))
            inner_id += 1
    except:
        sys.exit(-1)
    return sens;


if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'E:\\labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'E:\\testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "E:\\unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,
     test["review"].size, unlabeled_train["review"].size )



    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



    # ****** Split the labeled and unlabeled training sets into clean sentences
    #
    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    i=0
    #for review in train["review"]:
    review =  "Anirudh is a good boy"
    sentences += [gensim.models.doc2vec.LabeledSentence(KaggleWord2VecUtility.review_to_sentences(review, tokenizer)[0],labels = ['sent_%s' %i])]
    #i+=1
    
# 
#     print "Parsing sentences from unlabeled set"
#     for review in unlabeled_train["review"]:
#         sentences += [gensim.models.doc2vec.LabeledSentence(KaggleWord2VecUtility.review_to_sentences(review, tokenizer)[0],labels = ['sent_%s' %i])]
#         i+=1

#     print type(sentences)
    
#     with open("E:\\try.txt", 'wb') as f:
#         pickle.dump(sentences, f)
#         
#     with open("E:\\try.txt", 'rb') as f:
#         sentences = pickle.load(f)
    
    # ****** Set parameters and train the word2vec model
    #
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
#     num_features = 600    # Word vector dimensionality
#     min_word_count = 40   # Minimum word count
#     num_workers = 4       # Number of threads to run in parallel
#     context = 30          # Context window size
#     downsampling = 1e-8   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    print "here"
    
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    print "here"
    model.build_vocab(sentences)
    print "here"
    model.train(sentences)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    print "here"
    model.init_sims(replace=True)
    print "here"
    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    print "here"
    model.save(model_name)
    print "here"


    try:
    # ****** Create average vectors for the training and test sets

        train_sen_len=len(model.vocab)
        print "Printing mode: ", train_sen_len
        sentences=MyLabeledLineSentence(train_sen_len, model,['anirudh','is']) #getCleanReviews("anirudh is here"))
        model.train_labels=True
#         print "here"
        model.train_words=False
        print 'Sentences : ', sentences
        model.train(sentences)
    
        for sen in sentences:
            label = sen.labels[0]
            similar_array = model.most_similar(label)
            print "Input test sentence:%s\n" % (' '.join(sen.words).encode('utf-8'))
            for sim in similar_array:
                print ("\t\t%20s\t%.6f\n" % (sim[0].encode('utf-8'), sim[1]))
            print "\n"
            
            print "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
    except:
        print "haiiiiiiiiiiiiiiiiiiiiiiiiiii"
    
    
    
# positive    234234    mdghfdb \n
# cgfjkghjhfkmjnvbhmnkbn
# positive    234234    mdghfdb    
    
#     #
#     print "Creating average feature vecs for training reviews"
# 
#     trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
# 
#     print "Creating average feature vecs for test reviews"
# 
#     testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )
# 
# 
#     # ****** Fit a random forestto the training set, then make predictions
#     #
#     # Fit a random forest to the training data, using 100 trees
# 
#     
#     model = LR()
#     model.fit(trainDataVecs, train["sentiment"])
#     pr_teX = model.predict_proba(testDataVecs)[:, 1]
#     
# 
#     # Write the test results
#     output = pd.DataFrame( data={"id":test["id"], "sentiment":pr_teX} )
#     output.to_csv( "E:\\Word2Vec_AverageVectors.csv", index=False, quoting=3 )
#     print "Wrote Word2Vec_AverageVectors.csv"