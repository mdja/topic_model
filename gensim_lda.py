# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import nltk 

from nltk.corpus import stopwords
from sklearn.externals import joblib 

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def get_datawords(is_load =True, is_saved=False):
    if (is_load):
        datafile = 'processed_papers.csv'
        raw_data = pd.read_csv(datafile)


        reindexed_data = raw_data['clean_content']
        reindexed_data.index = raw_data['id']

        data_words = []
        for i in range(reindexed_data.shape[0]):
    #for i in range(2):
            data_words.append(reindexed_data.iloc[i].split())
        if (is_saved):
            joblib.dump(data_words,'data_words.dat')
    else:
        data_words = joblib.load('data_words.dat')
    return data_words

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def compute_performace(lda_model, text_data, corpus, dictionary):
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    

nltk.download('punkt')

stop_words = stopwords.words('english')
data_words = get_datawords()
# Build the bigram and trigram models
bigram = gensim.models.phrases.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.phrases.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
pprint(trigram_mod[bigram_mod[data_words[0]]])
# Form Bigrams
data_words_bigrams = make_bigrams(data_words, bigram_mod)

#nlp = spacy.load('en', disable=['parser', 'ner'])
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:1])
# Create Dictionary
dictionary = corpora.Dictionary(data_words_bigrams)
joblib.dump(dictionary,'dictionary.dat')

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [dictionary.doc2bow(text) for text in texts]
joblib.dump(corpus,'corpus.dat')
# View
#print(corpus[:1])
# Human readable format of corpus (term-frequency)
[[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]]

is_load_model = False
if (not is_load_model):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    joblib.dump(lda_model, 'lda_model_gensim.dat')
else:
    lda_model = joblib.load('lda_model_gensim.dat')
    
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#compute_performace(lda_model, data_words_bigrams, corpus, dictionary)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
vis    