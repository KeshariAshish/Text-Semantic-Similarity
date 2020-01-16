import csv
import nltk
nltk.download('stopwords')
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 
data = pd.read_csv("Text_Similarity_Dataset.csv") 

# print(data.head())
def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id

# from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
# stop_words = PlaintextCorpusReader(r'/home/harry/nltk_data/corpora/reuters2', 'stopwords').words()

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import LineTokenizer
tk = LineTokenizer(blanklines ='keep')
i = 0
for index, row in data.iterrows():
     # access data using column names
    #  print(index, row['Unique_ID'], row['text1'], row['text2'])
    token1 = tk.tokenize(row['text1'])
    token2 = tk.tokenize(row['text2'])
    #  print(token1)
     
    t1= []
    for i in range(len(token1)):
        words = token1[i].split()
        for r in words:
            if not r in stop_words:
                t1.append(r)
    t2= []
    for i in range(len(token2)):
        words = token2[i].split()
        for r in words:
            if not r in stop_words:
                t2.append(r)

    # word_to_id1 = mapping(t1)
    # word_to_id2 = mapping(t2)
    t3 = set()
    t3 = t1+t2
    # t3.append(t1)

    # t3 = set(t3)
    word_to_id = dict()
    for i, token in enumerate(set(t3)):
        word_to_id[token] = i
    # print(word_to_id)    
    v1 = [[0 for i in range(len(t3))]]
    v2 = [[0 for i in range(len(t3))]]

    for w in t1:
        if w in t3:
            v1[0][word_to_id[w]] = 1
    for w in t2:
        if w in t3:
            v2[0][word_to_id[w]] = 1
    # from sklearn.metrics.pairwise import cosine_similarity 
    
    ans = cosine_similarity(v1, v2)
    i=0
    with open('output.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([index,ans[0][i]]) 
        i = i + 1                     
    
    # break        
        
    