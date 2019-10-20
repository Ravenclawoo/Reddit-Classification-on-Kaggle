from nltk import FreqDist
from collections import Counter
from sklearn.model_selection import StratifiedKFold
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

import os
import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


#input: raw_train
#output: X and y
# document = pd.DataFrame({'Document': corpus, 'Category': labels})
# document = document[['Document', 'Category']]
# print(document)



start = time.time()

#some tools and list to use in the future
tokenizer = nltk.WordPunctTokenizer()
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
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
		# sentence = remove_nonenglish(sentence)
		# words lemmatize
		sentence = words_lemmatization(sentence)
		preprocessed.append(sentence)	 
	return preprocessed



path = os.path.abspath(os.path.dirname(sys.argv[0]))


#读reddit_train.csv
comments = pd.read_csv(path+'/reddit_train.csv', usecols=[1], encoding='Latin-1')
comments = np.array(comments)

corpus = []
for i in range(0, len(comments)):
	corpus.extend(comments[i].astype(str))


#读all-y-numbers.csv
subreddits = pd.read_csv(path+'/all-y-numbers.csv', usecols=[1])
subreddits = np.array(subreddits)

labels = []
for i in range(0, len(subreddits)):
	labels.extend(subreddits[i].astype(str))           # training labels
labels = np.array(labels)

#读reddit_test.csv
test_comments = pd.read_csv(path+'/reddit_test.csv',usecols=[1], encoding='Latin-1')
test_comments = np.array(test_comments)

test_corpus = []
for i in range(0, len(test_comments)):
	test_corpus.extend(test_comments[i].astype(str))


train_x = data_preprocess(corpus)
test_x = data_preprocess(test_corpus)


preprocessed_corpus = train_x
print(len(preprocessed_corpus))



c_vector = TfidfVectorizer(max_features=40000, binary=False)
c_vector.fit_transform(preprocessed_corpus)
words_bag = c_vector.get_feature_names()
print(len(words_bag))
print(len(preprocessed_corpus))
feature_matrix_train = c_vector.transform(train_x)     # training features
feature_matrix_test = c_vector.transform(test_x)       # testing features

#print(type(feature_matrix_train))
#print(type(feature_matrix_test))
# print(labels)


###########################################################################
###########################################################################
###########################################################################

#直接用生成的3个东西：feature_matrix_train, labels, feature_matrix_test
# sklearn的模型直接fit(feature_matrix_train, labels), 然后predict(feature_matrix_test)


# five-fold cross validation
skf = StratifiedKFold(n_splits=5)
for train_index, validation_index in skf.split(feature_matrix_train, labels):
	features_train, features_validation = feature_matrix_train[train_index], feature_matrix_train[validation_index]
	labels_train, labels_validation = labels[train_index], labels[validation_index]
	print("Training sample class distribution: ", features_train.shape[0])
	print("Validation sample class distribution: ", features_validation.shape[0])
	# Support Vector Machine (SVC) classifier
	clf = SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr', tol=1e-3)
	clf.fit(features_train, labels_train)
	print("Training accuracy is:", clf.score(features_train, labels_train))
	print("Training is finished, ready to perform validation.")
	labels_validation_predict = clf.predict(features_validation)
	print("Validation accuracy is: ", clf.score(features_validation, labels_validation))


# compute the running time
end = time.time()
print("The running time is:", (end-start))




