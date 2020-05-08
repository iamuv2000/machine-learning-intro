#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:44:06 2020

@author: yuvrajsingh
"""


#Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t' , quoting = 3)
#delimiter specifies we are using a 'tsv' file
#quoting = 3 -> Ignores all quotation marks in the file


# Cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for n in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][n]) 
    #First param -> characters we do not want to remove ; Replace those characters with ' '
    review = review.lower() #Make all char lowercase
    review = review.split() #Generate a list of the words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #we are using set to increase the speed, as it is faster to search in a 'set' than a list
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1100) #1500 most frequent words only
X = cv.fit_transform(corpus).toarray()
y = dataset['Liked']

#Using Naiive bais

#Splitting data into training and test set
from sklearn.model_selection import train_test_split 
X_train , X_test, y_train, y_test = train_test_split(X , y, test_size = 0.40 , random_state = 0)

# Fitting Naiive Bais to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
text = "Nice place will come again!"
#Cleaning new text
review = re.sub('[^a-zA-Z]',' ',text)
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus.append(review)
 
#Creating the sparse matrix again
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X1 = cv.fit_transform(corpus).toarray()
 
#Predicting the outcome
y_pred = classifier.predict(X[len(X)-1:len(X1)])
#y_pred = classifier.predict(X_test)


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred) #First arg is the actual values, the second is our predictions
#cm




