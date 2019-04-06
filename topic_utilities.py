# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""

import numpy as np
from scipy.sparse.csr import csr_matrix
from collections import Counter


# Define helper functions
# First we develop a list of the top words used across all one million paper_texts, giving us a glimpse into the core vocabulary of the source data. Stop words are omitted here to avoid any trivial conjunctions, prepositions, etc.
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''returns a tuple of the top n words in a sample and their accompanying counts, given a CountVectorizer object and text sample'''
    
    #print(text_data.head())
    vectorized_paper_texts = count_vectorizer.fit_transform(text_data.as_matrix().astype('U'))
    #print(vectorized_paper_texts)
    
    vectorized_total = np.sum(vectorized_paper_texts, axis=0)
    #print(vectorized_total)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    #print(word_indices)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    #print(word_values)
    word_vectors = np.zeros((n_top_words, vectorized_paper_texts.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1
    #print(word_vectors)
    words = [word[0].encode('ascii').decode('utf-8') for word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])

# Define helper functions
def get_keys(topic_matrix):
    '''returns an integer list of predicted topic categories for a given topic matrix'''
    keys = []
    for i in range(topic_matrix.shape[0]):
        keys.append(topic_matrix[i].argmax())
    return keys

def keys_to_counts(keys):
    '''returns a tuple of topic categories and their accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


def get_top_n_words_topics(n, n_topics,  keys, document_term_matrix, count_vectorizer):
    '''returns a list of n_topic strings, where each string contains the n most common 
        words in a predicted category, in order'''
    #print("n topics = {0}, keys= {1}".format(n_topics,keys))
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
#                print("{0}, {1},{2}".format(i, keys[i], topic))
                temp_vector_sum += document_term_matrix[i]
        #print("temp_vector_sum = {0}".format(temp_vector_sum))
        if (isinstance(temp_vector_sum, csr_matrix)):
            temp_vector_sum = temp_vector_sum.toarray()
            top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
            top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words

# Define helper functions
def get_mean_topic_vectors(n_topics, keys, two_dim_vectors):
    '''returns a list of centroid vectors from each predicted topic category'''
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])    
        if len(articles_in_that_topic) > 0:
            articles_in_that_topic = np.vstack(articles_in_that_topic)
            mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
            mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors