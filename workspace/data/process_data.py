import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Create a pandas dataframe by merging the messages and catgories
    data stored at input paths"""
    
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge them using the common id
    df = messages.merge(categories,on="id")
    return df


def clean_data(df):
    """ 
    Perform following steps for cleaning input data:
     1- create a dataframe of with individual categories as columns
     2- convert category values to just numbers 0 or 1
     3- drop duplicate rows
    """
    
    # Step 1
    categories = df["categories"].str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda strin: strin[:-2]).to_list()
    categories.columns = category_colnames
    
    # Step 2
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).\
                        apply(lambda strin: strin[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop(columns=['categories'],inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories],axis=1)
    
    # Step 3
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """ Save the input dataframe into a table named disaster_message_categories
    in an sqlite database """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_message_categories', engine, index=False)
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()