# dataprocessing
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# language processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# misc
from tqdm import tqdm
from time import time


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


@timer_func
def remove_stopwords(dataset):
    # downloads stopwords
    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join(
        [word for word in word_tokenize(x) if not word in stop_words]))


def get_dataset(name):
    df = pd.read_csv(f"datasets/{name}")

    # delete unnamed column with ids
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # rename
    df.columns = ['label', 'text', 'class']
    return df


def dataset_info(dataset):
    # show class (im)balance
    print("Dataset overview: ")
    info = dataset.groupby('label')['class'].count().reset_index()
    info.rename(columns={'class': 'count'}, inplace=True)
    info['percentage'] = 100 * info['count'] / info['count'].sum()
    print(info)


def predict_message_class(lr, vector):
    text = input('Enter Text(Subject of the mail): ')
    stop_words = set(stopwords.words('english'))
    text = [' '.join([word for word in word_tokenize(text)
                     if not word in stop_words])]
    term = vector.transform(text).toarray()
    print('Predicted Class:', end=' ')
    print('Spam' if lr.predict(term)[0] else 'Not Spam')
    prob = lr.predict_proba(term)*100
    print(f"Not Spam: {prob[0][0]}%\nSpam: {prob[0][1]}%")


def main():
    # prepare dataset from csv
    dataset = get_dataset("spam_ham_dataset.csv")

    # print (im)balance
    dataset_info(dataset)

    # remove stopwords - words that bring no meaning to the text (etc. I, is, are, the)
    remove_stopwords(dataset)

    # prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.loc[:, 'text'], dataset.loc[:, 'class'], test_size=0.30, random_state=1337)

    # preprocess text to build ML model
    vector = CountVectorizer()
    vector.fit(X_train)
    print('Number of Tokens: ', len(vector.vocabulary_.keys()))

    # prepare document term matrix
    X_train = vector.transform(X_train).toarray()
    print(f"Number of Observations before: {X_train.shape[0]}")

    #SMOTE
    #sm = SMOTE(random_state=1337)
    #X_train, y_train = sm.fit_resample(X_train, y_train)
    #print(f"Number of Observations after: {X_train.shape[0]}")

    #Undersampling
    #rus = RandomUnderSampler(random_state=42)
    #X_train, y_train = rus.fit_resample(X_train, y_train)
    #print(f"Number of Observations after: {X_train.shape[0]}")


    #Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print(f"Number of Observations after: {X_train.shape[0]}")

    # use model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # test
    X_test = vector.transform(X_test).toarray()
    pred = clf.predict(X_test)
    print(f"F1 Score: {round(f1_score(y_test, pred) * 100, 2)}%")

    predict_message_class(clf, vector)


if __name__ == "__main__":
    main()
