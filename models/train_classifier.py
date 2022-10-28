import sys
import pandas as pd
import numpy as np
import re
import joblib

import os
import sqlite3 as db

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Eventually not necessary
from sklearn.metrics import classification_report
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])

def load_data(database_filepath):
    '''
    Function: load the database
    Input: filepath of sql database
    Output: messages list, category list and category names list
    '''
    
    # Windows use, see https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
    # engine = create_engine('sqlite:///C:\\path\\to\\foo.db')
    
    path = os.path.dirname(os.path.realpath(__file__)) # directory path of the app
    con = db.connect(os.path.join(path, database_filepath))
    df = pd.read_sql_query("SELECT * from messages", con)
    
    con.close()
    
    #df = pd.read_sql_table('messages', engine)
    
    #engine = create_engine('sqlite:///..\\Data\\'+database_filename)
    
    
    #df = pd.read_sql_table('messages', 'sqlite:///'+database_filepath)

    
    X = df['message'].values
    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    Y = df[category_names].values
    return X, Y, category_names

def tokenize(text):
    '''
    Fuction: tokenize the message
    Input: text to tokenize
    Output: list of tokens
    '''
    text = text.lower() # Convert all leters to low case.
    text = re.sub(r"[^a-z0-9]", " ", text) # Anything that isn't a through z or 0 through 9 will be replaced by a space
    tokens = word_tokenize(text) # tokenize text
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')] # lemmatize and remove stop words
    return tokens


def build_model():
    '''
    Fuction: build model optimization grid
        In the function there is a pipeline with two transformers to vectorize the words in the messages, and a classifier that help to predict the relevant categories.
        In order to optimize the model, a  list with the most important parameter to check for better performance.
    Input: none
    Output: optimization grid of the model.
    '''
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tdidf_transformation', TfidfTransformer(smooth_idf = False)),
        ('clasifier', MultiOutputClassifier(estimator=RandomForestClassifier()))
        ])
        
    parameters = {
    'clasifier__estimator__criterion': ['gini', 'entropy', 'log_loss'],
    'clasifier__estimator__n_estimators': [10, 50, 100, 200]
    }
    model_grid = GridSearchCV(pipeline, param_grid=parameters)
    return model_grid

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Fuction: Model evaluation and print of prediction performance for each category.
    Input: model, a list of messages, a list of correct clasifications, and the list of category names
    Output: print the performance of the model on the given data.
    '''
    Y_pred = model.predict(X_test)
    df_ypred = pd.DataFrame(Y_pred, columns= category_names)
    df_ytest= pd.DataFrame(Y_test, columns= category_names)
    for column in list(df_ypred):
        print (column)
        print (classification_report(df_ytest[column].tolist(), df_ypred[column].tolist()))
    return


def save_model(model, model_filepath):
    '''
    Fuction: save the best model in a pickle formal file.
    Input: model and file path for the pickle file.
    Output: file saved with the model.
    '''
    joblib.dump(model.best_estimator_, model_filepath, compress = 1)
    return


def main():
    '''
    Fuction: main function interfacing with the command line for training the classifier
    Input: 2 arguments: filepath of the messages database and desired filename of the best achived model during training. 
    Output: file saved with the best model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()