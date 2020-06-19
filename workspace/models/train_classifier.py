import sys
import pandas as pd
from sqlalchemy import create_engine
import time
import pickle

# for tokenizing
import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# for feature extraction and modeling
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Read the input database to create and return separate numpy arrays for
    messages, category values, and names of categories respectively
    """
    
    # creating sqlite engine to interact with database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("disaster_message_categories",engine)
    
    # separating training and labels data and list of category names
    X = df.message.values
    Y_df = df[['related', 'request', 'offer', 'aid_related', 'medical_help', \
            'medical_products', 'search_and_rescue', 'security', 'military',\
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', \
            'missing_people', 'refugees', 'death', 'other_aid', \
            'infrastructure_related', 'transport', 'buildings', 'electricity'\
            , 'tools', 'hospitals', 'shops', 'aid_centers', \
            'other_infrastructure', 'weather_related', 'floods', 'storm', \
            'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'\
           ]]
    Y = Y_df.values
    category_names=Y_df.columns
    
    return X,Y,category_names

def tokenize(text):
    """
    Perform following steps for tokeninzing input text string: 
    - get list of all urls using regex
    - replace each url in text string with placeholder
    - tokenize text using nltk
    - for each token, lemmatize, normalize case, and remove leading/trailing white space

    """

    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Create a ML pipeline with a tokenizer, a feature constructor,
    and a multi output classifier with either Random Forest or
    XGBoost. 
    
    Commenting RandomForest pipeline because XGBoost pipeline is 
    quicker in training. 
    """
    # create pipeline with desired transformers and ML methods
#     pipeline = Pipeline([
#                 ('vect', CountVectorizer(tokenizer=tokenize,
#                                         max_df=0.75,
#                                         ngram_range=(1, 2))),
#                 ('tfidf', TfidfTransformer()),                
#                 ('clf', MultiOutputClassifier(\
#                     RandomForestClassifier(max_features=500,
#                                            n_estimators=100))
#                 )
#             ])
    
    # alternative for faster training
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),                
                ('clf', MultiOutputClassifier(\
                   XGBClassifier(max_depth = 4,
                                n_estimators = 100,
                                min_child_weight = 1,
                                gamma = 1,
                                subsample = 1.0,
                                colsample_bytree = 1.0)
                ))
            ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Using input testing data X_test and groundtruth Y_test:
    - Report f1 score, precision and recall of input model 
    for each output category.
    - Report overall acuuracy of the model. 
    """
    
    # make predictions for test set
    y_pred = model.predict(X_test)
    
    for i in range(0,len(Y_test[0,:])):
        print(classification_report(Y_test[:,i], y_pred[:,i]))
    
    print("...............................................................")
    print("Overall Accuracy: %.3f" %((y_pred == Y_test).mean().mean()))
    
    return None
    
def save_model(model, model_filepath):
    """ save the model at a desired location """
    pickle.dump(model, open(model_filepath, 'wb'))
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time() 
        model.fit(X_train, Y_train)
        print("--- Training finished in %s seconds ---" % (time.time() - start_time))
        
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