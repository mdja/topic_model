# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:19:31 2019

@author: lenovo pc
"""

import numpy as np
import pandas as pd
import topic_utilities
import generate_colormap
#from IPython.display import display
#from tqdm import tqdm

# abstract syntax tree
import ast


import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

from textblob import TextBlob
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label

'''
Display most used words in corpus
'''
def display_top_n_words(text_data, n_top_words=10):   
    count_vectorizer = CountVectorizer(stop_words='english')
    words, word_values = topic_utilities.get_top_n_words(n_top_words, count_vectorizer, text_data)
    
    fig, ax = plt.subplots(figsize=(30,8))
    ax.bar(range(len(words)), word_values)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words)
    ax.set_title('Top Words')

'''
Generate a histogram of paper word lengths, and use part-of-speech tagging to understand 
the types of words used across the corpus. This requires first converting all paper text strings to TextBlobs 
and calling the ```pos_tags``` method on each, yielding a list of tagged words for each paper_text.
A complete list of such word tags is available [here](https://www.clips.uantwerpen.be/pages/MBSP-tags).
'''
def pos_tag_word_papers(text_data):
    while True:
        try:
            tagged_paper_text = pd.read_csv('papers-pos-tagged.csv', index_col=0)
            word_counts = []
            pos_counts = {}
    
            for paper_text in tagged_paper_text[u'tags']:
                paper_text = ast.literal_eval(paper_text)
                word_counts.append(len(paper_text))
                for tag in paper_text:
                    if tag[1] in pos_counts:
                        pos_counts[tag[1]] += 1
                    else:
                        pos_counts[tag[1]] = 1
    
        except IOError:
            tagged_paper_text = [TextBlob(text_data.iloc[i]).pos_tags for i in range(text_data.shape[0])]
    
            tagged_paper_text = pd.DataFrame({'tags':tagged_paper_text})
            tagged_paper_text.to_csv('papers-pos-tagged.csv')
            continue
        break

    print('Jumlah total kata dalam paper (tidak termasuk stop words dll):', np.sum(word_counts))
    print('Rata-rata jumlah kata dalam paper (tidak termasuk stop words dll): ', np.mean(word_counts))
    
    
    pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
    pos_sorted_counts = sorted(pos_counts.values(), reverse=True)
    
    fig, ax = plt.subplots(figsize=(18,8))
    ax.bar(range(len(pos_counts)), pos_sorted_counts)
    ax.set_xticks(range(len(pos_counts)))
    ax.set_xticklabels(pos_sorted_types)
    ax.set_title('Part-of-Speech Tagging for NIPS Papers Corpus')
    ax.set_xlabel('Type of Word')

'''
display topic category vs count 
'''
def display_topics_bar(n_topics, title, vectorizer, document_term_matrix, topic_matrix):
    
    keys = topic_utilities.get_keys(topic_matrix)
    categories, counts = topic_utilities.keys_to_counts(keys)
    top_n_topic = topic_utilities.get_top_n_words_topics(30, n_topics, keys, document_term_matrix, vectorizer)
    
    for i in range(len(top_n_topic)):
        print("Topic {}: ".format(i), top_n_topic[i])
        
    top_3_word_topics = topic_utilities.get_top_n_words_topics(3, n_topics, keys, document_term_matrix, vectorizer)
    labels = ['Topic {}: \n'.format(i) + top_3_word_topics[i] for i in range(len(categories))]
    
    for i in range(len(top_3_word_topics)):
        print("Topic {}: ".format(i), top_3_word_topics[i])
        
    fig, ax = plt.subplots(figsize=(80,8))
    ax.bar(categories, counts)
    ax.set_xticks(categories)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    
'''
display low dimensional projection of topic vector in documents
'''
def display_tsne(output_fn, title, keys, mean_topic_vectors, top_3_topic, tsne_vectors, colormap):
    output_file(output_fn)

    
    plot = figure(title=title.format(n_topics), plot_width=800, plot_height=800)
    plot.scatter(x=tsne_vectors[:,0], y=tsne_vectors[:,1], color=colormap[keys])
    
    for t in range(len(mean_topic_vectors)):
    #    print(t)
        label = Label(x=mean_topic_vectors[t][0], y=mean_topic_vectors[t][1], 
                      text=top_3_topic[t], text_color=colormap[t])
        plot.add_layout(label)
        
    show(plot)


datafile = 'processed_papers.csv'
raw_data = pd.read_csv(datafile)

reindexed_data = raw_data['clean_content']
reindexed_data.index = raw_data['id']
n_topics = 50
matlib_colormap = generate_colormap.rand_cmap(n_topics, type='bright', first_color_black=False, last_color_black=False, verbose=True)
colormap = generate_colormap.convert_to_bokeh_colormap(matlib_colormap, n_topics)

############### DISPLAY WORD STATISTICS ###########################
#display_top_n_words(reindexed_data, 10)
#pos_tag_word_papers(reindexed_data)

############### DISPLAY TOPICS ###########################
###############   LSA #########################
tfid_vectorizer = joblib.load('tfidf_vectorizer.dat')
document_term_matrix_tfidf = joblib.load('document_term_matrix_tfidf.dat')
lsa_topic_matrix = joblib.load('lsa_topic_matrix.dat')
title = 'LSA Topic Category Counts'
display_topics_bar(n_topics, title, tfid_vectorizer, document_term_matrix_tfidf,lsa_topic_matrix)

tsne_model = joblib.load('tsne_lsa_model.dat')
tsne_vectors = joblib.load('tsne_lsa_vectors.dat')

lsa_keys = topic_utilities.get_keys(lsa_topic_matrix)
top_3__word_topics = topic_utilities.get_top_n_words_topics(3, n_topics, lsa_keys, document_term_matrix_tfidf, tfid_vectorizer)
mean_topic_vectors = topic_utilities.get_mean_topic_vectors(n_topics, lsa_keys, tsne_vectors)
output_fn = 'ouputlsatsne.html'
title="t-SNE Clustering of {} LSA Topics"
display_tsne(output_fn, title, lsa_keys, mean_topic_vectors, top_3__word_topics, tsne_vectors, colormap)

###############   LDA #########################

#count_vectorizer = joblib.load('count_vectorizer.dat')
#document_term_matrix_count = joblib.load('document_term_matrix_count.dat')
#lda_topic_matrix = joblib.load('lda_topic_matrix.dat')
#title = 'LDA Topic Category Counts'
#display_topics_bar(n_topics, title, count_vectorizer, document_term_matrix_count,lda_topic_matrix)
#
#tsne_model = joblib.load('tsne_lda_model.dat')
#tsne_vectors = joblib.load('tsne_lda_vectors.dat')
#
#lda_keys = topic_utilities.get_keys(lda_topic_matrix)
#top_3__word_topics = topic_utilities.get_top_n_words_topics(3, n_topics, lda_keys, document_term_matrix_count, count_vectorizer)
#mean_topic_vectors = topic_utilities.get_mean_topic_vectors(n_topics, lsa_keys, tsne_vectors)
#output_fn = 'ouputldatsne.html'
#
#title="t-SNE Clustering of {} LDA Topics"
#display_tsne(output_fn, title, lsa_keys, mean_topic_vectors, top_3__word_topics, tsne_vectors, colormap)