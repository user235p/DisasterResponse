# Software for clasification of disaster messages

## Installations

The code is written in Python.
You need some libraries:

- Pandas
- Sys
- sqlalchemy


You need to download the disaster message files from 8t8
The data is available at https://i?????????

Here you have the link to the files we need:
https://info.????????

Extract please the following two files from the zip and put them into the data directory:

**disaster_categories.csv**

**disaster_messages.csv**

## Project Motivation

With this project, it is possible to load the train a disaster message clasifier in order to help organisations to find relevant information when receiving messages as a dissaster occurs. For example from tweeter or sms sources.

For this purpose you can find an easy program structure to load and train the clasifier with the data provided from 8T8.
With the trained model, it is possible to use a web interface to let the help organizations introduce the incomming messages, and then show the relevant related labels.

## File Descriptions

The code is at the following python script files:
**process_data.py**

**train_classifier.py**

In oder to run it, you need the two .csv files mentioned before.

## How to Interact with your project

Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## Licensing, Authors, Acknowledgements

Please feel free to use my code.
The template comes from the udacity curs data scientist.

Thanks for 8T8 for giving the disaster text messages free, so every one out there can achieve a better clasifier and help where a new dissaster happens.



