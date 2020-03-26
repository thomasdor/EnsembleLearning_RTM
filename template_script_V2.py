"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) ...
        (2) ...  
        etc.
"""

"""
    Import necessary packages
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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#To visualise the whole row in a df.head()
pd.set_option('display.max_colwidth', -1)

X_train = pd.read_csv('/Users/thomasdorveaux/Desktop/Ensemble/X_train_update.csv')
y_train = pd.read_csv('/Users/thomasdorveaux/Desktop/Ensemble/Y_train_CVw08PX.csv')
X_test = pd.read_csv('/Users/thomasdorveaux/Desktop/Ensemble/X_test_update.csv')


X_train.head()
X_train.drop(X_train[['description','productid','imageid']],axis=1, inplace=True)
X_test.drop(X_test[['description','productid','imageid']],axis=1, inplace=True)

##################   Binding x_train and y_train  #####################
X_train['y_train']=y_train['prdtypecode']


################# Preprocessing
X_train['designation'] = X_train['designation'].str.lower()

def remove_characters(string):
    string = re.sub("([^\w]|[\d_])+", " ",  string)
    return string

X_train['designation'] = X_train['designation'].apply(remove_characters)

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

#import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#create a TfidfVectorizer object
tfidf = TfidfVectorizer()

#Vectorize the sample text
X_train_set = tfidf.fit_transform(X_train_prepro['designation_final'])

print("Shape of the TF-IDF Matrix:")
print(X_train_set.shape)
print("TF-IDF Matrix:")
"""
        etc.
"""

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
def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators = 100, max_depth=2, random_state=0) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    """
       This is just an example, plese change as necceary. Just maintain final output format with proper names of the models as described above.
    """
    model_1_acc, model_1_f1 = run_model_1(...)
    model_2_acc, model_2_f1 = run_model_2(...)
    """
        etc.
    """

    # print the results
    print("model_1", model_1_acc, model_1_f1)
    print("model_2", model_2_acc, model_2_f1)
    """
        etc.
    """
