import numpy as np
import pandas as pd
import topic_utilities
#from IPython.display import display
#from tqdm import tqdm

# abstract syntax tree
import ast


import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD #LSA
from sklearn.decomposition import LatentDirichletAllocation #LDA
from sklearn.manifold import TSNE

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec_model import LDA2Vec

from sklearn.externals import joblib

from textblob import TextBlob
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label

import nltk


nltk.download('punkt')

datafile = 'processed_papers.csv'
raw_data = pd.read_csv(datafile)


reindexed_data = raw_data['clean_content']
reindexed_data.index = raw_data['id']

count_vectorizer = CountVectorizer(stop_words='english')
n_top_words=30
text_data=reindexed_data
words, word_values = topic_utilities.get_top_n_words(n_top_words, count_vectorizer, text_data)

fig, ax = plt.subplots(figsize=(30,8))
ax.bar(range(len(words)), word_values)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_title('Top Words')

# Next we generate a histogram of paper word lengths, and use part-of-speech tagging to understand 
# the types of words used across the corpus. This requires first converting all paper text strings to TextBlobs 
# and calling the ```pos_tags``` method on each, yielding a list of tagged words for each paper_text.
# A complete list of such word tags is available [here](https://www.clips.uantwerpen.be/pages/MBSP-tags).
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
        tagged_paper_text = [TextBlob(reindexed_data.iloc[i]).pos_tags for i in range(reindexed_data.shape[0])]

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

##################################################################
##################### TOPIC MODELLING  ###########################
##################################################################


tfid_vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
joblib.dump(tfid_vectorizer, 'tfidf_vectorizer.dat')
#tfid_vectorizer = CountVectorizer(stop_words='english', max_features=50000)
#text_sample = reindexed_data.sample(n=len(reindexed_data), random_state=0).as_matrix()
#print('papers before tfidf vectorization: ', reindexed_data.iloc[123])
document_term_matrix_tfidf = tfid_vectorizer.fit_transform(reindexed_data)
#print('papers after tfidf vectorization: \n', document_term_matrix_tfidf[123])
n_topics = 50

################ 1. Latent Semantic Analysis ####################

lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix_tfidf)
lsa_keys = topic_utilities.get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = topic_utilities.keys_to_counts(lsa_keys)

top_n_topic_lsa = topic_utilities.get_top_n_words_topics(30, n_topics, lsa_keys, document_term_matrix_tfidf, tfid_vectorizer)

for i in range(len(top_n_topic_lsa)):
    print("Topic {}: ".format(i), top_n_topic_lsa[i])
    
top_3_topic_lsa = topic_utilities.get_top_n_words_topics(3, n_topics, lsa_keys, document_term_matrix_tfidf, tfid_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_topic_lsa[i] for i in range(len(lsa_categories))]

for i in range(len(top_3_topic_lsa)):
    print("Topic {}: ".format(i), top_3_topic_lsa[i])
    
fig, ax = plt.subplots(figsize=(80,8))
ax.bar(lsa_categories, lsa_counts)
ax.set_xticks(lsa_categories)
ax.set_xticklabels(labels)
ax.set_title('LSA Topic Category Counts')

######################
#### TSNE
######################

tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)

output_file('outputbokeh.html')


colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    "#902233", "#a65798", "#164560", "#3464bd", "#a52455",
    "#c2e567", "#f34612", "#e6d678", "#a3be5f", "#f2e150"
    ])
colormap = colormap[:n_topics]


top_3_topic_lsa = topic_utilities.get_top_n_words_topics(3, n_topics, lsa_keys, document_term_matrix_tfidf, tfid_vectorizer)
lsa_mean_topic_vectors = topic_utilities.get_mean_topic_vectors(n_topics, lsa_keys, tsne_lsa_vectors)

plot = figure(title="t-SNE Clustering of {} LSA Topics".format(n_topics), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])

for t in range(len(lsa_mean_topic_vectors)):
#    print(t)
    label = Label(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
                  text=top_3_topic_lsa[t], text_color=colormap[t])
    plot.add_layout(label)
    
show(plot)


################ 2. Latent Dirichlet Allocation ####################

count_vectorizer = CountVectorizer(stop_words='english', max_features=50000)
document_term_matrix_count = count_vectorizer.fit_transform(reindexed_data)
#print('papers after tfidf vectorization: \n', document_term_matrix_count[123])

lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(document_term_matrix_count)

lda_keys = topic_utilities.get_keys(lda_topic_matrix)
lda_categories, lda_counts = topic_utilities.keys_to_counts(lda_keys)

top_n_words_lda = topic_utilities.get_top_n_words_topics(n_topics, n_topics, lda_keys, document_term_matrix_count, count_vectorizer)

for i in range(len(top_n_words_lda)):
    print("Topic {}: ".format(i), top_n_words_lda[i])
    
top_3_words = topic_utilities.get_top_n_words_topics(3, n_topics, lda_keys, document_term_matrix_count, count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in range(len(lda_categories))]

for i in range(len(top_3_words)):
    print("Topic {}: ".format(i), top_3_words[i])
    
fig, ax = plt.subplots(figsize=(80,8))
ax.bar(lda_categories, lda_counts)
ax.set_xticks(lda_categories)
ax.set_xticklabels(labels)
ax.set_title('LDA Topic Category Counts')

#However, in order to properly compare LDA with LSA, we again take this topic matrix and project it into two dimensions with $t$-SNE.
tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, 
                        n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)


top_3_words_lda = topic_utilities.get_top_n_words_topics(3, n_topics, lda_keys, document_term_matrix_count, count_vectorizer)
lda_mean_topic_vectors = topic_utilities.get_mean_topic_vectors(n_topics, lda_keys, tsne_lda_vectors)

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), plot_width=700, plot_height=700)
plot.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], color=colormap[lda_keys])

for t in range(len(lda_mean_topic_vectors)):
    label = Label(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 
                  text=top_3_words_lda[t], text_color=colormap[t])
    plot.add_layout(label)

show(plot)

################ . 3. LDA 2 Vec ####################


