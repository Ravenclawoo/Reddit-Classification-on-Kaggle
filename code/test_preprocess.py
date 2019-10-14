from data_preprocess_feature_extraction import data_preprocess
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
import sys
import random
import nltk

import string
import numpy as np

import pandas as pd





path = os.path.abspath(os.path.dirname(sys.argv[0]))



#先用train.csv的目的是得到wordbag

#transfer each comment to a string, form a new array named "corpus"
comments = pd.read_csv(path+'/reddit_train.csv',usecols=[1])
comments = np.array(comments)

corpus = []
for i in range(0, len(comments)):
	corpus.extend(comments[i].astype(str))



preprocessed_corpus = data_preprocess(corpus)




#feature extraction
# c_vector = CountVectorizer(max_features=700)
c_vector = CountVectorizer(binary=True)
reddit_train_comment = c_vector.fit_transform(preprocessed_corpus)
words_bag = c_vector.get_feature_names()
print(len(words_bag))


#transfer each comment to a string, form a new array named "corpus_test"
comments_test = pd.read_csv(path+'/reddit_test.csv',usecols=[1])
comments_test = np.array(comments_test)

corpus_test = []
for i in range(0, len(comments_test)):
	corpus_test.extend(comments_test[i].astype(str))


print(len(corpus_test))
print(corpus_test[-2:])



preprocessed_corpus_test = data_preprocess(corpus_test)


print(preprocessed_corpus_test[-2:])





# c2_vector = c_vector.fit_transform(words_bag)

# print(c_vector.get_feature_names())

feature_matrix_test = c_vector.transform(preprocessed_corpus_test)
feature_matrix_test = feature_matrix_test.toarray()
print(feature_matrix_test[-1])



# feature_matrix_test = words_bag.fit_transform(preprocessed_corpus_test).toarray()
# print(len(feature_matrix_test[-1]))



#all_X_test is all the whole feature matrix in "reddit_test.csv"
all_X_test = pd.DataFrame(feature_matrix_test, columns=words_bag)
# print(all_X)





all_X_test.to_csv(path+'/no-b-20635-all-X-test.csv', sep=',', header=True, index=True)
# all_X_test.to_csv(path+'/fixed-b-1000-X-test.csv', sep=',', header=True, index=True)

print("feature matrix for test successful!")
