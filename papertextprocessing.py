# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 

  
class PaperTextProcessing():     
    url_pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'    
    url_pattern2 = r'https://t.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,}'
    digit_pattern = r'^\d+\s|\s\d+\s|\s\d+$'
    one_word_pattern = r'\s[a-zA-Z]{1}\s'
    additional_stopwords = ['et ', 'al', 'cc.', 'a', 'd', 'g', 'e', 'y', 'ga', 'gmn', 'tdk', 'nah', 'sih', 'blm', 'ni', 'di', 'sy', 'sya', 'rt', 'jl', 'jl.', 'jln', 'jln.', 'no', 'no.', 'dlm', 'tx', 'thx', 'he', 'd', 'k', 'sm']
    
    
    def __init__(self, paper_text=None):
        if (paper_text != None):
            paper_text.text = paper_text.strip()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords = self.stopwords.union(set(self.additional_stopwords))
        
    def set_paper_text(self, text):
        self.paper_text = text
        
    def get_paper_text(self):
        return self.paper_text
            
    
    def clean_up_text_url(self, text):
        text = re.sub(self.url_pattern, '', text)
        text = re.sub(self.one_word_pattern, '', text)
        text = text.replace("https://t.?", '')
        text = text.replace("https://t?", '')
        text = text.replace("https://?", '')
        return re.sub(self.url_pattern2, '', text)
    
    def clean_up_text_digits(self, text):
        #text = ''.join([i for i in str(text) if not i.isdigit()])
        #return text
        return re.sub(self.digit_pattern,'', text)
    
    def remove_stop_words(self, text): 
        tokens = word_tokenize(text)                
        filtered_sentence = [w for w in tokens if w.isalpha()] 
        filtered_sentence = [w for w in filtered_sentence if not w in self.stopwords] 

        return ' '.join(filtered_sentence)
    
    def lemmatize_text(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(tokens[i]) for i in range(len(tokens))]
        return ' '.join([tokens[i] for i in range(len(tokens))])
    
    def getPaperParts(self, judul):
        #print(judul)
        #paper = self.paper_text.replace('\n', ' ')
        paper = self.paper_text
        paper = re.sub(r'^\d+\s','',paper).strip()
        #find title
        # ambil dulu 1500 kata???
        pengarang = ''
        abstrak = ''
        isi = ''
        kesimpulan = ''
        referensi = ''
        # find author
        startIdx = len(judul)+1
        lastIdx = paper[:1000].upper().find('ABSTRACT')
        if (lastIdx == -1):
            return (pengarang, abstrak, isi, kesimpulan, referensi)
        pengarang = paper[startIdx:lastIdx].replace('*','').replace('\n',' ').strip()
#        print(judul)
#        print(pengarang)
        #find abstract
        startIdx = lastIdx + len('ABSTRACT')
#        check pattern INTRODUCTION atau 1. Introduction
        match = re.search('(1|1.)*\s*INTRODUCTION', paper[startIdx:2000].upper())
        if not match:
            match = re.search('\w*(1.)*[A|AN|THE]*\s[A-Z]{2}\w*[A-Z]\w*', paper[startIdx:2000])
            if match:
#                print('0. FOUND INTRO PART')
                lastIdx = startIdx + match.start()
            else:
                match = re.search('[\\n]+(1|1.)[\\n]+', paper[startIdx:2000])
                if match:
#                    print('1. FOUND INTRO PART')
                    lastIdx = startIdx + match.start()
                else:
#                    print('NOT FOUND Intro part!')
                    lastIdx = startIdx + 1000 #default max length of abstract
        else:
#            print(match.start())
            lastIdx = startIdx + match.start()
        if lastIdx != -1: 
#            print('0. startidx = {0}, lastIdx = {1}'.format(startIdx, lastIdx)) 
            abstrak = paper[startIdx:lastIdx].replace('\n', ' ').strip() 
#            print(abstrak)
            startIdx = lastIdx
            lastIdx = paper.lower().rfind('conclusion')
            
            if (lastIdx != -1 and lastIdx > startIdx):
#                print('1. startidx = {0}, lastIdx = {1}'.format(startIdx, lastIdx))                 
                isi = paper[startIdx:lastIdx].strip()
                startIdx = lastIdx + len('CONCLUSION') 
                lastIdx = paper.lower().rfind('references')
                kesimpulan = paper[startIdx:lastIdx].strip()
            else:
#                print('2. startidx = {0}, lastIdx = {1}'.format(startIdx, lastIdx))                                 
                lastIdx = paper.lower().rfind('references')
#                print('3. startidx = {0}, lastIdx = {1}'.format(startIdx, lastIdx))                                 
                isi = paper[startIdx:lastIdx].strip()
            if (lastIdx != -1 and lastIdx > startIdx):
                startIdx = lastIdx + len('REFERENCES')
#                print('4. startidx = {0}, lastIdx = {1}'.format(startIdx, lastIdx))                                 
                referensi = paper[startIdx:len(paper)].strip()
        return (pengarang, abstrak, isi, kesimpulan, referensi)

    def clean_up_text(self, text):
        #replace newline char with empty string
        formatted_text = text.replace('\n', ' ')
        formatted_text = re.sub(r'\b[A-z]{1,2}\b','', formatted_text)
        #remove trailing number if exist
        formatted_text = re.sub(r'^\d+\s','', formatted_text).strip()
        formatted_text = re.sub(r'^\[a-z]{1}\s','', formatted_text).strip()
        formatted_text = formatted_text.lower()
        formatted_text = self.clean_up_text_url(formatted_text)
        formatted_text = self.clean_up_text_digits(formatted_text)        
        formatted_text = formatted_text.replace(',',' ')
        formatted_text = formatted_text.replace('?','')
        formatted_text = formatted_text.replace('  ',' ')
        formatted_text = self.lemmatize_text(formatted_text)
        formatted_text = self.remove_stop_words(formatted_text)

        return formatted_text

 