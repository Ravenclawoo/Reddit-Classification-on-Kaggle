import numpy as np
import pandas as pd
import os
import sys
from keras.utils import to_categorical


path = os.path.abspath(os.path.dirname(sys.argv[0]))

#transfer each subreddit to a string, form a new array named "labels"
subreddits = pd.read_csv(path+'/reddit_train.csv',usecols=[2])
subreddits = np.array(subreddits)

labels = []
for i in range(0, len(subreddits)):
	if subreddits[i] == 'hockey':
		subreddits[i] = 0
	if subreddits[i] == 'nba':
		subreddits[i] = 1
	if subreddits[i] == 'leagueoflegends':
		subreddits[i] = 2
	if subreddits[i] == 'soccer':
		subreddits[i] = 3
	if subreddits[i] == 'funny':
		subreddits[i] = 4
	if subreddits[i] == 'movies':
		subreddits[i] = 5
	if subreddits[i] == 'anime':
		subreddits[i] = 6
	if subreddits[i] == 'Overwatch':
		subreddits[i] = 7
	if subreddits[i] == 'trees':
		subreddits[i] = 8
	if subreddits[i] == 'GlobalOffensive':
		subreddits[i] = 9
	if subreddits[i] == 'nfl':
		subreddits[i] = 10
	if subreddits[i] == 'AskReddit':
		subreddits[i] = 11
	if subreddits[i] == 'gameofthrones':
		subreddits[i] = 12
	if subreddits[i] == 'conspiracy':
		subreddits[i] = 13
	if subreddits[i] == 'worldnews':
		subreddits[i] = 14
	if subreddits[i] == 'wow':
		subreddits[i] = 15
	if subreddits[i] == 'europe':
		subreddits[i] = 16
	if subreddits[i] == 'canada':
		subreddits[i] = 17
	if subreddits[i] == 'Music':
		subreddits[i] = 18
	if subreddits[i] == 'baseball':
		subreddits[i] = 19
		
	labels.extend(subreddits[i])

# print(labels)
numerical_labels = pd.DataFrame(labels)

#save number labels
numerical_labels.to_csv(path+'/all-y-numbers.csv', sep=',', header=True, index=True)




one_hot_labels = to_categorical(numerical_labels)
print(one_hot_labels)


one_hot_labels = pd.DataFrame(one_hot_labels)

#save one hot labels
one_hot_labels.to_csv(path+'/all-y-one-hot.csv', sep=',', header=True, index=True)


