import json
import time
import benepar
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)

parser = benepar.Parser("benepar_en3")
VP_PHRASE, SBAR_PHRASE, S_PHRASE, THAT_PHRASE, IN_PHRASE = "VP", "SBAR", "S", "that", "IN"


def calculate_word_relative_frequency(word, sentences):
    """
    :param word: The word for which the relative frequency will be calculated
    :param sentences: The list of sentences from which the relative frequency will be calculated
    :return: the relative frequency of the word
    """

    word_counter, total_words = 0, 0

    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        word_counter += words.count(word)

    return round(100 * (word_counter / total_words), 3)


def calculate_relative_frequencies(discriminative_words, explicit_list, implicit_list):
    """
    :param discriminative_words: list with the discriminative words
    :param explicit_list: list of explicit sentences
    :param implicit_list: list of implicit sentences
    :return: tuple of 2 lists - list of the relative frequency for each word according to the explicit list
     and list of the relative frequency for each word according to the implicit list
    """

    # Creating Empty lists for each category of relative frequency - explicit and implicit
    explicit_relative_frequency, implicit_relative_frequency = [], []

    # Calculating the relative frequency for each word
    for discriminative_word in discriminative_words:
        explicit_relative_frequency.append(calculate_word_relative_frequency(discriminative_word, explicit_list))
        implicit_relative_frequency.append(calculate_word_relative_frequency(discriminative_word, implicit_list))

    return explicit_relative_frequency, implicit_relative_frequency


def get_discriminative20(explicit_set, implicit_set):
    """
    :param explicit_set: a set of explicit sentences
    :param implicit_set: a set of implicit sentences
    :return: None
    """

    # Initializing tfidf vectorizer object
    tfidf_vect = TfidfVectorizer(stop_words="english")

    # Creating dataframe of sentences and labels (0 - implicit, 1 - explicit)
    explicit_list, implicit_list = list(explicit_set), list(implicit_set)
    sentences = explicit_list + implicit_list
    labels = [1] * len(explicit_list) + [0] * len(implicit_list)
    df = pd.DataFrame({'sentences': sentences, 'labels': labels})

    # Converting sentences to numerical TfIdf vectors
    features_data = tfidf_vect.fit_transform(df["sentences"])

    # 20 words that have the highest discriminative power
    discriminative20 = SelectKBest(score_func=chi2, k=20)
    discriminative_words_indices = discriminative20.fit(features_data, df["labels"]).get_support(indices=True)
    discriminative_words = tfidf_vect.get_feature_names_out()[discriminative_words_indices]

    # Calculating relative frequencies for all discriminative words
    explicit_relative_frequency, implicit_relative_frequency = calculate_relative_frequencies(discriminative_words, \
                                                                                              explicit_list, implicit_list)

    # Creating a list of labels.
    # Each label in the list describes the label to which the high relative frequency of each word belongs
    labels_list = []
    for freq_imp, freq_exp in zip(implicit_relative_frequency, explicit_relative_frequency):
        labels_list.append("Implicit") if freq_imp > freq_exp else labels_list.append("Explicit")

    # Concentrating of the data in the data frame object
    df_relative_frequency = pd.DataFrame({"Word": discriminative_words, \
                                          "Explicit relative freq": explicit_relative_frequency, \
                                          "Implicit relative freq": implicit_relative_frequency, \
                                          "Label": labels_list})

    # Printing the results
    print(df_relative_frequency)


def identify_explicit_and_implicit_that_clauses(filename):

    print(f'looking for explicit and implicit "that" usages in {filename}')

    # Creating Empty sets for each category - explicit and implicit
    explicit_set, implicit_set = set(), set()

    # List of the parts of speech we want to find in the sentence when we are in Verb Phrase constituent
    verb_pos = ["VB", "VBP", "VBD", "VBZ", "VBG", "VBN"]

    # Reading the sentences from file to list
    with open(filename, 'r', encoding="utf-8") as fin:
        sentences = fin.read().splitlines()

    # The implementation of logic to classify the sentences into explicit or implicit
    for sentence in sentences:
        tree = parser.parse(sentence)
        for sub_tree in tree.subtrees():
            if sub_tree.label() == VP_PHRASE and len(sub_tree) == 2:
                if sub_tree[1].label() == SBAR_PHRASE and sub_tree[0].label() in verb_pos:
                    if len(sub_tree[1]) == 2 and sub_tree[1][1].label() == S_PHRASE\
                      and sub_tree[1][0].label() == IN_PHRASE and sub_tree[1][0][0] == THAT_PHRASE:
                        explicit_set.add(sentence)
                    elif len(sub_tree[1]) == 1 and sub_tree[1][0].label() == S_PHRASE:
                        implicit_set.add(sentence)

    # Finding the 20 most descriptive words and their relative frequency
    #get_discriminative20(explicit_set, implicit_set)

    return explicit_set, implicit_set


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    explicit, implicit = identify_explicit_and_implicit_that_clauses(config['input_filename'])
    print(f'found {len(explicit)} explicit, and {len(implicit)} implicit cases')

    with open(config['explicit_filename'], 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(explicit))
    with open(config['implicit_filename'], 'w', encoding="utf-8") as fout:
        fout.write('\n'.join(implicit))

    print(f'total time: {round(time.time() - start, 0)} sec')
