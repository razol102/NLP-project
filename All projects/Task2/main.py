import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2


def get_discriminative15(tfidf_vect, train_data_X, train_data_y):
    """
    :param tfidf_vect: tfidf vectorizer object
    :param train_data_X: numerical review vectors of tfidf representation
    :param train_data_y: reviews ranking labels
    :return: None
    """
    discriminative15 = SelectKBest(score_func=chi2, k=15)
    discriminative_words_indices = discriminative15.fit(train_data_X, train_data_y).get_support(indices=True)
    discriminative_words = tfidf_vect.get_feature_names_out()[discriminative_words_indices]
    print("The 15 words that have the highest discriminative power are:", discriminative_words)


def classify(train_file, test_file):

    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # Initializing tfidf vectorizer object
    tfidf_vect = TfidfVectorizer(ngram_range=(1,2))

    # Reading json files to data frames
    train_data, test_data = pd.read_json(train_file, lines=True), pd.read_json(test_file, lines=True)

    # Concatenating the "reviewTest" and the "summary" columns
    train_data["CombinedText"] = train_data["reviewText"].fillna(str()) + " " + train_data["summary"].fillna(str())
    test_data["CombinedText"] = test_data["reviewText"].fillna(str()) + " " + test_data["summary"].fillna(str())

    # Filtering only the relevant columns
    train_data, test_data = train_data[["CombinedText", "overall"]], test_data[["CombinedText", "overall"]]

    # Shuffling the train and test data
    train_data, test_data = train_data.sample(frac=1), test_data.sample(frac=1)

    # Converting reviews to numerical TfIdf vectors
    train_data_X, train_data_y = tfidf_vect.fit_transform(train_data["CombinedText"]), train_data["overall"]
    test_data_X, test_data_y = tfidf_vect.transform(test_data["CombinedText"]), np.array(test_data["overall"])

    # 15 words that have the highest discriminative power
    #get_discriminative15(tfidf_vect, train_data_X, train_data_y)

    # Running Logistic Regression Model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_data_X, train_data_y)
    pred = clf.predict(test_data_X)

    # Calculation of metrics
    cm = confusion_matrix(test_data_y, pred)
    #print(cm)
    f1s = f1_score(test_data_y, pred, average=None)
    acc = accuracy_score(test_data_y, pred)

    test_results = {'class_1_F1': f1s[0],
                    'class_2_F1': f1s[1],
                    'class_3_F1': f1s[2],
                    'class_4_F1': f1s[3],
                    'class_5_F1': f1s[4],
                    'accuracy': acc}
    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
