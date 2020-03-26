#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 02:49:13 2020

@author: thomasdorveaux
"""

"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Michaël Allouche
        (2) Raphaël Attali
        (3) Thomas Dorveaux
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
from spacy.tokens import Token
import re
import fr_core_news_sm
import en_core_web_sm
import en_core_web_sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
sns.set()

abspath = os.path.abspath('template_script_V2.py')
dname = os.path.dirname(abspath)
os.chdir(dname)

os.chdir(sys.path[0])
os.getcwd()


pd.set_option('display.max_colwidth', -1)


# Load the datasets
X_train = pd.read_csv('/Users/thomasdorveaux/Desktop/EnsembleLearning_RTM/Data/X_train_update.csv')
y_train = pd.read_csv('/Users/thomasdorveaux/Desktop/EnsembleLearning_RTM/Data/Y_train_CVw08PX.csv')
X_test = pd.read_csv('/Users/thomasdorveaux/Desktop/EnsembleLearning_RTM/Data/X_test_update.csv')

X_train.drop(X_train[['description','productid','imageid']],axis=1, inplace=True)
X_test.drop(X_test[['description','productid','imageid']],axis=1, inplace=True)

##################   Binding x_train and y_train  #####################
X_train['y_train']=y_train['prdtypecode']

#################    Preprocessing    #####################

# lowercase strings
X_train['designation'] = X_train['designation'].str.lower()


# remove non aplha numeric characters
def remove_characters(string):
    string = re.sub("([^\w]|[\d_])+", " ",  string)
    return string

X_train['designation'] = X_train['designation'].apply(remove_characters)

# define language detectors 
language_detector = LanguageDetector()
nlp_fr = fr_core_news_sm.load(disable=["tagger", "parser","ner","entity_linker","textcat","entity_ruler","sentencizer","merge_noun_chunks","merge_entities","merge_subtokens"])
nlp_fr.add_pipe(nlp_fr.create_pipe('sentencizer'))
nlp_fr.add_pipe(language_detector)


# add a column for languages
X_train['language'] = X_train['designation'].str[:].apply(lambda row : nlp_fr(row)._.language['language'])


# plot the different languages of the dataset
fig, axes = plt.subplots(1, 1, figsize = (10,5))

ax = sns.countplot(x="language", 
                   data=X_train,
                   order=['fr','en','it','ca']
                     )
ax.set(xlabel='language', ylabel='Number of products', title='Number of products per language')

for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), 
    (p.get_x() + p.get_width() / 2., p.get_height()),
    ha='center', va='center', fontsize=11, color='gray', xytext=(0, 4),
    textcoords='offset points')
    
plt.show()

# plot the differentproduct categories
fig, axes = plt.subplots(1, 1, figsize = (10,5))

ax = sns.countplot(x="y_train", 
                   data=X_train
                   #order=['fr','en','it','ca']
                     )
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Product type code', ylabel='Number of products', title='Number of products per product type code')

plt.show()

# determine the top languages
X_train['language'].value_counts().head()

# tokenizing
nlp_fr = fr_core_news_sm.load(disable=["tagger", "parser","ner","entity_linker","textcat","entity_ruler","sentencizer","merge_noun_chunks","merge_entities","merge_subtokens"])
nlp_en = en_core_web_sm.load(disable=["tagger", "parser","ner","entity_linker","textcat","entity_ruler","sentencizer","merge_noun_chunks","merge_entities","merge_subtokens"])
languages_model={'fr':nlp_fr,'en':nlp_en}


# define support functions
def tokenize(row):
    return [token.orth_ for token in row]

def recombining_tokens_into_a_string(list_of_tokens):
    return " ".join(list_of_tokens)

# define preprocess main function
def preprocess(language_model,X_train):
  X_train_final=X_train[~(X_train['language'].isin(list(languages_model.keys())))]
  X_train_final['designation_final']=X_train_final['designation']
  for language in languages_model.keys():
    X_train_lang=X_train.loc[X_train['language'] == language ] # subset the dataframe for the specific language
    spacy_tokens = X_train_lang['designation'].apply(languages_model[language]) #Define spacy model
    #####token_lang = spacy_tokens.apply(tokenize) # Tokenize
    token_lang=spacy_tokens.apply(lambda x: [token.lemma_ for token in x if not token.is_stop]) # Extract lemmas and remove stop words
    X_train_lang['designation_final'] = token_lang.apply(recombining_tokens_into_a_string)
    X_train_final=pd.concat([X_train_final,X_train_lang],axis=0)
  return X_train_final

X_train_prepro = preprocess(languages_model,X_train)

# withdraw accents
def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string


X_train_prepro['designation_final'] = X_train_prepro['designation_final'].apply(normalize_accent)

#create a TfidfVectorizer object
tfidf = TfidfVectorizer()

#Vectorize the sample text
X_train_set = tfidf.fit_transform(X_train_prepro['designation_final'])

print("Shape of the TF-IDF Matrix:")
print(X_train_set.shape)
print("TF-IDF Matrix:")


"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""

# splitting the data into train and validation sets
y_train.drop(['Unnamed: 0'],axis = 1, inplace = True)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_set, y_train, test_size=0.2, random_state=42)

    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
################ Defining the classifier function ##############
### two remarks here :
    #1) The results for the accuracy and the F1 score will be different here
    # as in the report, a cross-validation method is used with cv = 5.
    #2) the hyper-paramaters chosen here have been found using
    # gridsearch on a jupyter file that run on google cloud.
    # We decided not to include the grid search in this file as 
    # it was quite consuming. 
    
def decision_tree(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 150,min_samples_leaf=1,random_state = 0)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")
    return rf_accuracy, rf_f1

def ada_boost(X_train, y_train, X_test, y_test):
    clf = AdaBoostClassifier(random_state = 50, n_estimators = 60) 
    clf.fit(X_train, y_train.values.ravel())
    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")
    return rf_accuracy, rf_f1

def model_random_forest(X_train, y_train, X_test, y_test):

    clf = RandomForestClassifier(n_estimators = 140, max_depth = None,n_jobs=-1, random_state=0) 
    clf.fit(X_train, y_train.values.ravel())

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


def Xgboost(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(max_depth = 4,n_estimators = 50 ) 
    clf.fit(X_train, y_train.values.ravel())
    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")
    return rf_accuracy, rf_f1

def Naive_Bayes(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train.values.ravel())
    y_predicted = clf.predict(X_test)
    print(y_predicted)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")
    return rf_accuracy, rf_f1


"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":

    decisiontree_acc, decisiontree_f1 = decision_tree(X_train, y_train, X_valid, y_valid)
    ada_boost_acc, ada_boost_f1 = ada_boost(X_train,y_train ,X_valid , y_valid)
    randomforest_acc, randomforest_f1 = model_random_forest(X_train,y_train ,X_valid , y_valid )
    xgboost_acc, xgboost_f1 =Xgboost(X_train,y_train ,X_valid , y_valid)


    # print the results
    print("decision tree", decisiontree_acc, decisiontree_f1)
    print("adaboost", ada_boost_acc, ada_boost_f1)
    print("randomforest", randomforest_acc, randomforest_f1)
    print("xgboost", xgboost_acc, xgboost_f1)
