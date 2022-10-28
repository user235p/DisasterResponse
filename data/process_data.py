import sys
import pandas as pd
import sqlite3 as db

def load_data(messages_filepath, categories_filepath):
    '''
    Function: Load data from csv files, and merge them generating one column for each category.
    Input: names of both messages and categories files.
    Output: dataframe without merged data
    '''
    messages = pd.read_csv(messages_filepath) # Load messages table
    categories = pd.read_csv(categories_filepath) # Load categories table
    df = pd.merge(messages, categories, on='id') # Merge both tables based on id
    categories_separated = df['categories'].str.split(';', expand=True) # New dataframe extrated from the column categories

    # based on the first row, generate the names of the columns for dataframe categories_separated
    row = df.categories[0]
    category_colnames = [colname[:-2] for colname in row.split(';')]
    categories_separated.columns = category_colnames
    for column in categories_separated:
        # set each value to be the last character of the string
        categories_separated[column].astype(str)
        categories_separated[column] = categories_separated[column].str[-1].astype(int) # take the last character as information and ignore the rest.

    # Integrate categories_separated in dataframe df
    df.drop(columns='categories', inplace=True) # Eliminate original column categories
    df = pd.concat([df, categories_separated], axis=1) # Add the new separated columns with the data
    return df

def clean_data(df):
    '''
    Function: remove duplicates on the dataframe
    Input: dataframe
    Output: dataframe without duplicates
    '''
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Function: save dataframe into a sql database
    Input: dataframe and filename of the database
    Output: none.
    '''
    database = database_filename
    conn = db.connect(database)
    df.to_sql(name='messages', con=conn)
    conn.close()
    return



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