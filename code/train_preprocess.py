from data_preprocess_feature_extraction import data_preprocess
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import sys
import random
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import numpy as np




import pandas as pd
import re




path = os.path.abspath(os.path.dirname(sys.argv[0]))


#transfer each comment to a string, form a new array named "corpus"
comments = pd.read_csv(path+'/reddit_train.csv',usecols=[1])
comments = np.array(comments)

corpus = []
for i in range(0, len(comments)):
	corpus.extend(comments[i].astype(str))


#transfer each subreddit to a string, form a new array named "labels"
subreddits = pd.read_csv(path+'/reddit_train.csv',usecols=[2])
subreddits = np.array(subreddits)

labels = []
for i in range(0, len(subreddits)):
	labels.extend(subreddits[i].astype(str))






preprocessed_corpus = data_preprocess(corpus)

# print(preprocessed_corpus)






#feature extraction
# c_vector = CountVectorizer(max_features=700)
c_vector = CountVectorizer(binary=False)
reddit_train_comment = c_vector.fit_transform(preprocessed_corpus)
words_bag = c_vector.get_feature_names()
print(len(words_bag))

print(len(preprocessed_corpus))


feature_matrix = c_vector.fit_transform(preprocessed_corpus).toarray()
# print(feature_matrix)




#all_X is all the whole feature matrix in "reddit_train.csv"
all_X = pd.DataFrame(feature_matrix, columns=words_bag)
# print(all_X)

#all_y is all the labels in "reddit_train.csv"
# all_y = pd.read_csv(path+'/reddit_train.csv',usecols=[2])





# all_X.to_csv(path+'/b-1000-all-X.csv', sep=',', header=True, index=True)
# all_X.to_csv(path+'/no-b-1000-all-X-train.csv', sep=',', header=True, index=True)
all_X.to_csv(path+'/no-b-20635-all-X-train.csv', sep=',', header=True, index=True)
# all_y.to_csv(path+'/all-y.csv', sep=',', header=True, index=True)

print("feature matrix for train finish!")



