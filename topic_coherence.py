# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:02:44 2019

@author: lenovo pc
"""

# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""

from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from sklearn.externals import joblib

def coherence(topics, corpus, dictionary):
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value
    print('Topics coherence values = {}'.format(coherence))

dictionary = joblib.load('dictionary.dat')
corpus = joblib.load('corpus.dat')

file = open("topic lsa.txt","r")
#Repeat for each song in the text file
topics =[]
for line in file:
  
  #Let's split the line into an array called "fields" using the ";" as a separator:
  fields = line.split(":")
#  print(fields[1].strip())
  tokens = word_tokenize(fields[1])
  topics.append(tokens)

coherence(topics, corpus, dictionary)
file = open("topic nmf.txt","r")
#Repeat for each song in the text file
topics =[]
for line in file:
  
  #Let's split the line into an array called "fields" using the ";" as a separator:
  fields = line.split(":")
#  print(fields[1].strip())
  tokens = word_tokenize(fields[1])
  topics.append(tokens)

#pprint(topics)
coherence(topics, corpus, dictionary)

file = open("topic lda.txt","r")
#Repeat for each song in the text file
topics =[]
for line in file:
  
  #Let's split the line into an array called "fields" using the ";" as a separator:
  fields = line.split(":")
#  print(fields[1].strip())
  tokens = word_tokenize(fields[1])
  topics.append(tokens)

#pprint(topics)
coherence(topics, corpus, dictionary)

