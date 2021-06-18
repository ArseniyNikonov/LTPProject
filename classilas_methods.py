import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from transformers import BertTokenizer

def join_function(sentence):
    return ' '.join(sentence)

def wordpiece_tokenizer(dataset):
    tokenized = []
    for data in dataset:
        tokenized.append(tokenizer.tokenize(data))
    return tokenized
# Uploading the data


test = pd.read_csv('data/test.txt',sep=";",header=None)
train = pd.read_csv('data/train.txt',sep=";",header=None)
val = pd.read_csv('data/val.txt',sep=";",header=None)
#Set random seed for replication purposes
np.random.seed(500)

# Splitting X and Y 
X_train = train.iloc[:,0] 
Y_train = train.iloc[:,1] 

X_test = test.iloc[:,0] 
Y_test = test.iloc[:,1] 

X_val = val.iloc[:,0] 
Y_val = val.iloc[:,1] 



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

X_train = wordpiece_tokenizer(X_train)
X_train = list(map(join_function,X_train))
X_val = wordpiece_tokenizer(X_val)
X_val = list(map(join_function,X_val))
X_test = wordpiece_tokenizer(X_test)
X_test = list(map(join_function,X_test))



X_train,Y_train = np.array(X_train),np.array(Y_train)
X_test,Y_test = np.array(X_test),np.array(Y_test)
X_val,Y_val = np.array(X_val),np.array(Y_val)

Encoder = LabelEncoder()
Y_train = Encoder.fit_transform(Y_train)
Y_val   = Encoder.transform(Y_val)
Y_test  = Encoder.transform(Y_test)



print("#train instances: {} #dev: {} #test: {}".format(len(X_train),len(X_val),len(X_test)))

print("vectorize data..")
vectorizer = CountVectorizer()

# classifier = Pipeline( [('vec', vectorizer),
#                         ('clf', LogisticRegression(max_iter=1000,class_weight='balanced'))] )

classifier = Pipeline( [('vec', vectorizer),
                        ('nb', naive_bayes.ComplementNB())] )
print("train Byess model..")

classifier.fit(X_train,Y_train)

Y_predicted_test = classifier.predict(X_test)
print('Accuracy score ',accuracy_score(Y_test, Y_predicted_test))

Y_test_inverse = Encoder.inverse_transform(Y_test)
Y_predicted_test = Encoder.inverse_transform(Y_predicted_test)
print(metrics.classification_report(Y_test_inverse, Y_predicted_test))


# Y_predicted_val = classifier.predict(X_val)
# print('Accuracy score ',accuracy_score(Y_val, Y_predicted_val))

# Y_val = Encoder.inverse_transform(Y_val)
# Y_predicted_val = Encoder.inverse_transform(Y_predicted_val)
# print(metrics.classification_report(Y_val, Y_predicted_val))

print("train Logistic regression model..")
classifier = Pipeline( [('vec', vectorizer),
                        ('clf', LogisticRegression(max_iter=1000,class_weight='balanced'))] )

classifier.fit(X_train,Y_train)

Y_predicted_test = classifier.predict(X_test)
print('Accuracy score ',accuracy_score(Y_test, Y_predicted_test))

Y_predicted_test = Encoder.inverse_transform(Y_predicted_test)
print(metrics.classification_report(Y_test_inverse, Y_predicted_test))