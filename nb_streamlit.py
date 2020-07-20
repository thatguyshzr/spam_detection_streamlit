import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes

def model():
	df= pd.read_csv('spam.csv', encoding='latin-1')
	df= df.dropna(axis= 'columns')

	cv = CountVectorizer(stop_words = 'english')
	X = cv.fit_transform(df['v2'])

	df['v1'] = df['v1'].map({'spam':1, 'ham':0})
	X_train, X_test, y_train, y_test = train_test_split(X, df['v1'],
		test_size=0.33, random_state=42)

	bayes = naive_bayes.MultinomialNB(alpha = 15.73001)
	bayes.fit(X_train, y_train)
	return cv, bayes