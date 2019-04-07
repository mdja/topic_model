# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""

#import numpy as np
import pandas as pd
import topic_utilities
#from IPython.display import display
#from tqdm import tqdm

# abstract syntax tree
#import ast


#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD #LSA
from sklearn.decomposition import LatentDirichletAllocation #LDA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
#from lda2vec import utils
#from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
#from lda2vec_model import LDA2Vec

from sklearn.externals import joblib

#from textblob import TextBlob
#from bokeh.plotting import figure, output_file, show
#from bokeh.models import Label

import nltk


nltk.download('punkt')

datafile = 'processed_papers.csv'
raw_data = pd.read_csv(datafile)


reindexed_data = raw_data['clean_content']
reindexed_data.index = raw_data['id']

#count_vectorizer = CountVectorizer(stop_words='english')
#n_top_words=30
#text_data=reindexed_data
#words, word_values = topic_utilities.get_top_n_words(n_top_words, count_vectorizer, text_data)
#
#fig, ax = plt.subplots(figsize=(30,8))
#ax.bar(range(len(words)), word_values)
#ax.set_xticks(range(len(words)))
#ax.set_xticklabels(words)
#ax.set_title('Top Words')

##################################################################
##################### TOPIC MODELLING  ###########################
##################################################################

################ 1. Latent Semantic Analysis ####################

tfid_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
#tfid_vectorizer = CountVectorizer(stop_words='english', max_features=50000)
#text_sample = reindexed_data.sample(n=len(reindexed_data), random_state=0).as_matrix()
#print('papers before tfidf vectorization: ', reindexed_data.iloc[123])
document_term_matrix_tfidf = tfid_vectorizer.fit_transform(reindexed_data)
#print('papers after tfidf vectorization: \n', document_term_matrix_tfidf[123])
n_topics = 20

lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix_tfidf)

joblib.dump(tfid_vectorizer, 'tfidf_vectorizer.dat')
joblib.dump(document_term_matrix_tfidf, 'document_term_matrix_tfidf.dat')
joblib.dump(lsa_model, 'lsa_model.dat')
joblib.dump(lsa_topic_matrix, 'lsa_topic_matrix.dat')

lsa_keys = topic_utilities.get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = topic_utilities.keys_to_counts(lsa_keys)


tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)

joblib.dump(tsne_lsa_model, 'tsne_lsa_model.dat')
joblib.dump(tsne_lsa_vectors, 'tsne_lsa_vectors.dat')


################ NMF ####################

count_vectorizer = CountVectorizer(analyzer='word', max_features=20000);
document_term_matrix_count = count_vectorizer.fit_transform(reindexed_data)
transformer = TfidfTransformer(smooth_idf=False);
document_term_matrix_tfidf = transformer.fit_transform(document_term_matrix_count);
document_term_matrix_tfidf_norm = normalize(document_term_matrix_tfidf, norm='l1', axis=1)
nmf_model = NMF(n_components=n_topics, init='nndsvd')
nmf_topic_matrix = nmf_model.fit_transform(document_term_matrix_tfidf_norm)

joblib.dump(count_vectorizer, 'count_vectorizer_nmf.dat')
joblib.dump(document_term_matrix_count, 'document_term_matrix_count_nmf.dat')
joblib.dump(document_term_matrix_tfidf_norm, 'document_term_matrix_tfidf_norm.dat')
joblib.dump(nmf_model, 'nmf_model.dat')
joblib.dump(nmf_topic_matrix, 'nmf_topic_matrix.dat')

tsne_nmf_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_nmf_vectors = tsne_nmf_model.fit_transform(nmf_topic_matrix)
joblib.dump(tsne_nmf_model, 'tsne_nmf_model.dat')
joblib.dump(tsne_nmf_vectors, 'tsne_nmf_vectors.dat')

################ 3. Latent Dirichlet Allocation ####################

count_vectorizer = CountVectorizer(stop_words='english', max_features=20000)
document_term_matrix_count = count_vectorizer.fit_transform(reindexed_data)
#print('papers after tfidf vectorization: \n', document_term_matrix_count[123])

lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0, learning_decay=0.9)
lda_topic_matrix = lda_model.fit_transform(document_term_matrix_count)


joblib.dump(count_vectorizer, 'count_vectorizer_lda.dat')
joblib.dump(document_term_matrix_count, 'document_term_matrix_count_lda.dat')
joblib.dump(lda_model, 'lda_model.dat')
joblib.dump(lda_topic_matrix, 'lda_topic_matrix.dat')


tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)

joblib.dump(tsne_lda_model, 'tsne_lda_model.dat')
joblib.dump(tsne_lda_vectors, 'tsne_lda_vectors.dat')

