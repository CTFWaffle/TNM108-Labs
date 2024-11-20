from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Building a pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
#print("MultinomialNB accuracy: ", np.mean(predicted == twenty_test.target))

# Training SVM classifier
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([
    ('vect', CountVectorizer()), 
    ('tfidf', TfidfTransformer()), 
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))
])

text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
#print("SVM accuracy: ", np.mean(predicted == twenty_test.target))

from sklearn import metrics
#print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))

#print(metrics.confusion_matrix(twenty_test.target, predicted))


from sklearn.model_selection import GridSearchCV
parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (0.02, 1e-5),
 'tfidf__sublinear_tf': (True, False),
 'vect__lowercase': (True, False),
 'vect__max_features': (None, 5000, 10000, 50000)
}
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],  
    'vect__min_df': [1, 2, 5],                     
    'vect__max_df': [0.9, 1.0],                   
    'vect__stop_words': [None, 'english'],       
    'vect__binary': [True, False],                
    'tfidf__use_idf': [True, False],              
    'tfidf__smooth_idf': [True, False],           
    'clf__alpha': [0.02, 0.01, 1e-5],             
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
#print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))