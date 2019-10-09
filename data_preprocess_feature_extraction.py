from nltk import FreqDist
from collections import Counter

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
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

from nltk.metrics import ConfusionMatrix
from six.moves import zip


import pandas as pd
import re






path = os.path.abspath(os.path.dirname(sys.argv[0]))


#transfer each comment to a string, form a new array named "corpus"
comments = pd.read_csv(path+'/reddit_train.csv',usecols=[1])
comments = np.array(comments)

corpus = []
for i in range(0, len(comments)):
	corpus.extend(comments[i].astype(str))

# print(corpus)



#transfer each subreddit to a string, form a new array named "labels"
subreddits = pd.read_csv(path+'/reddit_train.csv',usecols=[2])
subreddits = np.array(subreddits)

labels = []
for i in range(0, len(subreddits)):
	labels.extend(subreddits[i].astype(str))

# print(labels)







# document = pd.DataFrame({'Document': corpus, 'Category': labels})
# document = document[['Document', 'Category']]

# print(document)



#some tools and list to use in the future
tokenizer = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
english_words = set(nltk.corpus.words.words())
# print(english_words)




#remove special symbols and raw preprocess
def remove_abnormal(text):
	#remove whitespaces
	text = text.strip()
	#lower case
	text = text.lower()
	#remove numbers
	text = re.sub('\d+', '', text)
	text = re.sub('_', '', text)
	text = re.sub('\s+', ' ', text)
	#remove html links
	text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
	#replace special characters with ''
	text = re.sub('[^\w\s]', '', text)
	return text

#remove stop words
def remove_stopwords(text):
	tokens = tokenizer.tokenize(text)
	tokens = [token.strip() for token in tokens]
	text = [token for token in tokens if token.lower() not in stop_words]
	return " ".join(text)
	

#remove non-english words
def remove_nonenglish(text):
	tokens = tokenizer.tokenize(text)
	tokens = [token.strip() for token in tokens]
	text = [token for token in tokens if token.lower() in english_words]
	return " ".join(text)


#lemmatization
def words_lemmatization(text):
	tokens = tokenizer.tokenize(text)
	tokens = [lemmatizer.lemmatize(token.strip()) for token in tokens]
	text = [token for token in tokens]
	return " ".join(text)


def data_preprocess(corpus):
	preprocessed = []
	for sentence in corpus:
		# remove special character and normalize docs
		sentence = remove_abnormal(sentence)
		# remove stopwords 
		sentence = remove_stopwords(sentence)
		# remove non-english
		sentence = remove_nonenglish(sentence)
		# words lemmatize
		sentence = words_lemmatization(sentence)
		preprocessed.append(sentence)	 
	return preprocessed


preprocessed_corpus = data_preprocess(corpus)

# print(preprocessed_corpus)






#feature extraction
c_vector = CountVectorizer(max_features=1000)
reddit_train_comment = c_vector.fit_transform(preprocessed_corpus)
words_bag = c_vector.get_feature_names()
# print(words_bag)




feature_matrix = c_vector.fit_transform(preprocessed_corpus).toarray()
# print(feature_matrix)




#all_X is all the whole feature matrix in "reddit_train.csv"
all_X = pd.DataFrame(feature_matrix, columns=words_bag)
# print(all_X)

#all_y is all the labels in "reddit_train.csv"
all_y = pd.read_csv(path+'/reddit_train.csv',usecols=[2])





all_X.to_csv(path+'/1000-all-X.csv', sep=',', header=True, index=True)
all_y.to_csv(path+'/all-y.csv', sep=',', header=True, index=True)





