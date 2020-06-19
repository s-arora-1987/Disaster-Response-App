import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_message_categories', engine)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load model
model = joblib.load("../models/classifier.pkl")

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Plot 1
    # create a histogram of messages counts in each category, sorted in descending order 
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', \
            'medical_products', 'search_and_rescue', 'security', 'military',\
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', \
            'missing_people', 'refugees', 'death', 'other_aid', \
            'infrastructure_related', 'transport', 'buildings', 'electricity'\
            , 'tools', 'hospitals', 'shops', 'aid_centers', \
            'other_infrastructure', 'weather_related', 'floods', 'storm', \
            'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'\
           ]
    category_counts = (df[category_names].sum()/len(df)).sort_values(ascending = False)
    # retrieve the category name for descending order 
    category_sequence_changed = list(category_counts.index)
    
    # Plot 2
    # compute the number of messages in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # dividing the message count for each genre on the basis of association with aid
    aid_rel0 = df[df['aid_related']==0].groupby('genre').count()['message']
    aid_rel1 = df[df['aid_related']==1].groupby('genre').count()['message']

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_sequence_changed,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories So Far',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {

           'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name = 'Aid related'

                ),
                Bar(
                    x=genre_names,
                    y= aid_rel0,
                    name = 'Aid not related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }

        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()