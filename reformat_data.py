# -*- coding: utf-8 -*-
"""
by mdja, itb, 2019
"""

import pandas as pd
import papertextprocessing as pt
#from IPython.display import display
#from tqdm import tqdm
#from collections import Counter

import nltk
 
nltk.download('punkt')

datafile = 'data/papers.csv'
#datafile = 'processed_papers.csv'
raw_data = pd.read_csv(datafile)

#raw_data['processed_text']=raw_data['paper_text']
raw_data['author']=raw_data['abstract']
raw_data['content']=raw_data['abstract']
raw_data['content']=raw_data['abstract']
raw_data['conclusion']=raw_data['abstract']
raw_data['references']=raw_data['abstract']
raw_data['clean_content']=raw_data['abstract']

paper_processing = pt.PaperTextProcessing(None)
to_delete = []
for idx in range(len(raw_data['paper_text'])):
#idx = 5722
    paper_text = raw_data['paper_text'][idx]
    paper_processing.set_paper_text(paper_text)
    judul_paper = raw_data['title'][idx]
    (pengarang, abstrak, isi, kesimpulan, referensi) = paper_processing.getPaperParts(judul_paper)
    
    if (isi == ''):    
#        print('TIDAK ADA ISI! idx = {}'.format(idx))
        to_delete.append(idx)
    #    break
    else:
    #    print('PAPER_TEXT : ' + paper_text)
    #    print('JUDUL : ' + judul_paper)
    #    print('PENGARANG : ' + pengarang)
    #    print('ABSTRAK : ' + abstrak)
    #    print('ISI : ' + isi)
    #    print('KESIMPULAN : ' + kesimpulan)
    #    print('REFERENCES : ' + referensi)
        raw_data['author'][idx] = pengarang
        raw_data['abstract'][idx] = abstrak
        raw_data['content'][idx] = isi
        clean_content = paper_processing.clean_up_text(isi)
        raw_data['clean_content'][idx] = clean_content 
        raw_data['conclusion'][idx] = kesimpulan
        raw_data['references'][idx] = referensi
   
print(to_delete)
raw_data = raw_data.drop(columns='paper_text')
raw_data = raw_data.drop(to_delete, axis=0)
#raw_data = raw_data.iloc[:2]
raw_data.to_csv('processed_papers.csv', index=True)
 