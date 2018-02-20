import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 750, init = 'uniform', activation = 'relu', input_dim = 1500 ))

classifier.add(Dense(output_dim = int(750/2), init='uniform',activation = 'relu'))

classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid' ))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'] )

classifier.fit(X_train,y_train, epochs=50)


y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
