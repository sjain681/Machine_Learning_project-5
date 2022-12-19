import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import gensim
import sys
import numpy as np
import logging
import random
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

labels_as_numbers = pd.read_csv("ps5_tweets_labels_as_numbers.csv")
labels = pd.read_csv("ps5_tweets_labels.csv")
text = pd.read_csv("ps5_tweets_text.csv")

stop_words = set(stopwords.words('english'))

def preprocess_tweet_text(tweet):
    tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
#     ps = PorterStemmer()
#     stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
    
    return " ".join(lemma_words).lower()

processed_tweets = []
for i in text['Tweet']:
    tw = preprocess_tweet_text(i)
    processed_tweets.append(tw)

df = pd.DataFrame(processed_tweets, columns = ['Tweets'])
result = pd.concat([labels, df], axis=1)
result = result.drop(['Id'], axis=1)


X = result['Tweets']
y = result['Sentiment']

kf = KFold(n_splits=3)
kf.get_n_splits(X)
print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

# Create a cross-validation strategy
cv = StratifiedKFold(n_splits=12, random_state=42)

# Instantiate the classification model and visualizer
model = MultinomialNB()
visualizer = cross_val_score(X_train, y_train, cv=cv, scoring='f1_weighted')

#visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

NB_model = MultinomialNB()
NB_model.fit(X_train_cv, y_train)
y_predict_nb = NB_model.predict(X_test_cv)

print('Accuracy score: ', accuracy_score(y_test, y_predict_nb),'\n')
print('Precision score: ', precision_score(y_test, y_predict_nb, average='weighted'))
print('Precision score: ', precision_score(y_test, y_predict_nb, average='macro'))
print('Precision score: ', precision_score(y_test, y_predict_nb, average='micro'))
print('Recall score:    ', recall_score(y_test, y_predict_nb, average='weighted'))
print('Recall score:    ', recall_score(y_test, y_predict_nb, average='macro'))
print('Recall score:    ', recall_score(y_test, y_predict_nb, average='micro'))
print('f1_score:        ', f1_score(y_test, y_predict_nb, average='weighted'))
print('f1_score:        ', f1_score(y_test, y_predict_nb, average='macro'))
print('f1_score:        ', f1_score(y_test, y_predict_nb, average='micro'))

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train_cv, y_train)
y_predict_lr = LR_model.predict(X_test_cv)
print(accuracy_score(y_test, y_predict_lr))

cm = confusion_matrix(y_test, y_predict_nb)
print(cm)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, fmt='g',
xticklabels=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'],
            yticklabels=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
"""
word2vec = gensim.models.Word2Vec(X_train, size=100, window=5, min_count=10, workers=8)

word2vec_weight = word2vec.wv.vectors
vocab_size, embedding_size = word2vec_weight.shape
print("Vocab Size: ", vocab_size)
print("Embedding Size: ", embedding_size)
print(word2vec.wv.most_similar('bad', topn=3))

def word2token(word):
    try:
        return word2vec.wv.vocab[word].index
    except KeyError:
        return 0

def token2word(token):
    return word2vec.wv.index2word[token]

max_sequence_length = 200
drop_threshold = 10000

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                   output_dim=embedding_size,
                   weights=[word2vec_weight],
                   input_length=max_sequence_length,
                   mask_zero=True,
                   trainable=False))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=20,
                   validation_data=(X_test, y_test), verbose=1)
"""