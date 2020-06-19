## Disaster-Response-App

The purpose of this project is to create a web application that categorizes the messages/ tweets associated with real-life disasters. The motivation behind this applications is to have a faster anlaysis about what kind of help is needed by person sending message to disaster responders. 


### Contents and Purpose of different files. 
Two csv files have data containing messages/tweets during real-life disasters, and the categories messages can be divided in (aid related, search & rescue, water, medical help, .. etc). After some pre-processing in file workspace/data/process_data.py, that data is saved in a SQL database called DisasterResponse. The details of this code are in ETL_Pipeline_Preparation notebook. The code in file workspace/data/train_classifier.py uses saved data to train a multi output classifier and saves result in pickle format; a detailed code for training classifier and tuning hyperparameters is in ML_Pipeline_Preparation notebook. File workspace/app/run.py creates a Flask web app to visualize some insights about of dataset, and to predict the category of input message using the classifier trained in workspace/data/train_classifier.py

#### Directory structure of Disaster-Response-App/workspace:

- app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

- data

|- disaster_categories.csv  # data to process 

|- disaster_messages.csv  # data to process

|- process_data.py

|- DisasterResponse.db   # database to save clean data to

- models

|- train_classifier.py

|- classifier.pkl  # saved model 


## Instructions for setting up application

Install Python 3.x and following dependencies:
- NumPy
- Pandas
- Matplotlib
- Json
- Plotly
- Nltk
- Flask
- Sklearn
- Sqlalchemy
- Sys
- Re
- Pickle

## Instructions for running the application:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


#### Example insight about dataset
Histogram for messages so far in different categories

![](/Images/histogram.png)



