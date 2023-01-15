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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedKFold

# misc
from tqdm import tqdm
from time import time
from copy import copy
from threading import Thread


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
    # nltk.download('stopwords')
    # nltk.download('punkt')

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


class MockPreprocessor:
    def __init__(self, a):
        pass

    def fit_resample(self, a, b):
        return a, b


def perform_classification(X, y, scores, clf, name, preprocessor, cv, vector, position):
    for train_idx, test_idx, in tqdm(cv.split(X, y), total=cv.get_n_splits(), desc=name, position=position):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # preprocess text to build ML model
            vector.fit(X_train)

            # prepare document term matrix
            X_train = vector.transform(X_train).toarray()
            X_train, y_train = preprocessor.fit_resample(X_train, y_train)

            # use model
            clf.fit(X_train, y_train)

            # test
            X_test = vector.transform(X_test).toarray()
            pred = clf.predict(X_test)

            scores[name]['f1_score'].append(round(f1_score(y_test, pred) * 100, 2))
            scores[name]['accuracy'].append(round(accuracy_score(y_test, pred) * 100, 2))
            scores[name]['precision'].append(round(precision_score(y_test, pred) * 100, 2))
            scores[name]['recall'].append(round(recall_score(y_test, pred) * 100, 2))


def main():
    # prepare dataset from csv
    dataset = get_dataset("spam_ham_dataset.csv")

    # print (im)balance
    dataset_info(dataset)

    # remove stopwords - words that bring no meaning to the text (etc. I, is, are, the)
    remove_stopwords(dataset)

    X = dataset['text']
    y = dataset['class']
    clf = MultinomialNB()
    vector = CountVectorizer()
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2652124)

    preprocessors = {
        'Without': MockPreprocessor(1337),
        'SMOTE': SMOTE(random_state=1337),
        'Undersampling': RandomUnderSampler(random_state=1337),
        'Oversampling': RandomOverSampler(random_state=1337)
    }
    scores = {
        'Without': {'f1_score': [], 'accuracy': [], 'precision': [], 'recall': []},
        'SMOTE': {'f1_score': [], 'accuracy': [], 'precision': [], 'recall': []},
        'Undersampling': {'f1_score': [], 'accuracy': [], 'precision': [], 'recall': []},
        'Oversampling': {'f1_score': [], 'accuracy': [], 'precision': [], 'recall': []}
    }

    threads = []
    for name, preprocessor in preprocessors.items():
        thread = Thread(target=perform_classification, args=(X, y, scores, copy(clf), name, preprocessor, cv, copy(vector), len(threads)))
        threads.append(thread)
        thread.start()    
        #thread.join() # uncomment if having memory issues
    
    for thread in threads:
        thread.join()

    # calculate meanscores
    mean_scores = {
        'Without': {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0},
        'SMOTE': {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0},
        'Undersampling': {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0},
        'Oversampling': {'f1_score': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    }
    for key, value in scores.items():
        for measurements_key, measurements_values in value.items():
            mean_scores[key][measurements_key] = np.round(
                np.mean(measurements_values), 2)

    # print results
    print("Mean scores:", flush=True)
    for key, value in mean_scores.items():
        print(key, ": ")
        for measurements_key, measurements_value in value.items():
            print(f"{measurements_key} : {measurements_value}%")
        print(' ')


if __name__ == "__main__":
    main()
