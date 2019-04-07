# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


from sklearn.decomposition import LatentDirichletAllocation #LDA
from sklearn.model_selection import GridSearchCV


count_vectorizer = joblib.load('count_vectorizer_lda.dat')
document_term_matrix_count = joblib.load('document_term_matrix_count_lda.dat')
lda_topic_matrix = joblib.load('lda_topic_matrix.dat')
lda_model = joblib.load('lda_model.dat')

_load_data_ = False
model = None
if (not _load_data_):
    # Define Search Param
    search_params = {'n_components': [20, 25, 30], 'learning_decay': [.5, .7, .9]}
    
    # Init the Model
    lda_model = LatentDirichletAllocation()
    # Init Grid Search Class
    model = GridSearchCV(lda_model, param_grid=search_params)
    
    # Do the Grid Search
    model.fit(document_term_matrix_count)
else:
    model = joblib.load('model_gridsearch.dat')
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(document_term_matrix_count))


# Create Document - Topic Matrix
lda_output = best_lda_model.transform(document_term_matrix_count)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]

# index names
docnames = ["Doc" + str(i) for i in range(document_term_matrix_count.shape[0])]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics