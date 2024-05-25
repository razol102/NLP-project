import json
from collections import defaultdict
import nltk
import numpy as np
import random
import copy
EMPTY_SPACE = "__________"


def create_int_defaultdict():
    return defaultdict(int)


def read_file_to_list(file):
    res = list()
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin.readlines():
            res.append(line.strip())
    return res


def create_bigrams(corpus, lexicon, candidates):
    """
    :param corpus: set of words which the bigrams are based on
    :param lexicon: set of common words in English
    :param candidates: list of words to fill in the cloze
    :return: data stracture (dictionary of dictionaries) which holds the bigrams from the corpus
    """
    print("Start creating bigrams")
    bigrams = defaultdict(create_int_defaultdict)

    with open(corpus, 'r', encoding="utf-8") as fin:
        line = fin.readline()
        while line:
            if line.strip() != "":
                line = line.lower().split()
                if line[0] in candidates and line[0] in lexicon:  # first word in a sentence
                    bigrams["<s>"][line[0]] += 1
                if line[-1] in candidates and line[-1] in lexicon:  # last word in a sentence
                    bigrams[line[-1]]["</s>"] += 1
                for word1, word2 in zip(line[:-1], line[1:]):
                    in_candidate = word1 in candidates or word2 in candidates
                    word1_in_lex = word1 in lexicon
                    word2_in_lex = word2 in lexicon
                    if in_candidate and word1_in_lex and word2_in_lex:
                        bigrams[word1][word2] += 1
            line = fin.readline()
    return bigrams


def create_trigrams(corpus, lexicon, candidates):
    """
    :param corpus: set of words which the bigrams are based on
    :param lexicon: set of common words in English
    :param candidates: list of words to fill in the cloze
    :return: data stracture (dictionary of dictionaries) which holds the bigrams from the corpus
    """
    print("Start creating trigrams")
    trigrams = defaultdict(create_int_defaultdict)

    with open(corpus, 'r', encoding="utf-8") as fin:
        line = fin.readline()
        while line:
            if line.strip() != "":
                line = line.lower().split()
                if len(line) > 2:  # more than 2 words in a sentence
                    in_candidate = line[0] in candidates or line[1] in candidates
                    if in_candidate and line[0] in lexicon and line[1] in lexicon:  # first 2 words in a sentence
                        trigrams[("<s>", line[0])][line[1]] += 1

                    in_candidate = line[-2] in candidates or line[-1] in candidates
                    if in_candidate and line[-2] in lexicon and line[-1] in lexicon:  # last 2 words in a sentence
                        trigrams[(line[-2], line[-1])]["</s>"] += 1

                    for word1, word2, word3 in zip(line[:-2], line[1:-1], line[2:]):
                        in_candidate = word1 in candidates or word2 in candidates or word3 in candidates
                        word1_in_lex = word1 in lexicon
                        word2_in_lex = word2 in lexicon
                        word3_in_lex = word3 in lexicon
                        if in_candidate and word1_in_lex and word2_in_lex and word3_in_lex:
                            trigrams[(word1, word2)][word3] += 1

                elif len(line) == 2:  # exactly 2 words in a sentence
                    in_candidate = line[0] in candidates or line[1] in candidates
                    word1_in_lex = line[0] in lexicon
                    word2_in_lex = line[1] in lexicon
                    if in_candidate and word1_in_lex and word2_in_lex:
                        trigrams[("<s>", line[0])][line[1]] += 1
                        trigrams[(line[0], line[1])]["</s>"] += 1

                else:  # only 1 word in a sentence
                    if line[0] in candidates and line[0] in lexicon:
                        trigrams[("<s>", line[0])]["</s>"] += 1
            line = fin.readline()

    return trigrams


def get_prob(candidate, bigrams, trigrams, sentence, i, lexicon_size):
    """
    :param candidate: a word that is a candidate to fill the empty space in the cloze
    :param trigrams: dictionary which holds the trigrams from the corpus
    :param bigrams: data stracture (dictionary of dictionaries) which holds the bigrams from the corpus
    :param sentence: sentence from the cloze
    :param i: the index of the candidate in the current sentence
    :param lexicon_size: num of words in the lexicon
    :return: probability of the candidate to fill the specific empty space in the cloze
    """
    k = 0.001
    return ((trigrams[(sentence[i - 2], sentence[i - 1])][candidate] + k) / (bigrams[sentence[i - 2]][sentence[i - 1]] + k * lexicon_size))\
            * ((trigrams[(sentence[i - 1], candidate)][sentence[i + 1]] + k) / (bigrams[sentence[i - 1]][candidate] + k * lexicon_size))\
            * ((trigrams[(candidate, sentence[i + 1])][sentence[i + 2]] + k) / (bigrams[candidate][sentence[i + 1]] + k * lexicon_size))


def compute_chance_accuracy(candidates, res):
    i = 0
    for word1, word2 in zip(candidates, res):
        if word1 == word2:
            i += 1
    return i / len(res)


def compute_mean_random_chance_accuracy(candidates):
    copied_candidates = copy.deepcopy(candidates)
    mean_random_chance_accuracy = 0

    for i in range(100):
        random.shuffle(copied_candidates)
        mean_random_chance_accuracy += compute_chance_accuracy(candidates, copied_candidates)

    mean_random_chance_accuracy /= 100

    return mean_random_chance_accuracy


def choose_candidates(input, candidates, bigrams, trigrams, lexicon):
    """
    :param input: the cloze file name
    :param candidates: list of candidates to fill the cloze
    :param bigrams: data stracture (dictionary of dictionaries) which holds the bigrams from the corpus
    :param trigrams: dictionary which holds the trigrams from the corpus
    :param lexicon: set of common words in English
    :return: a list of words to fill the empty spaces in the cloze respectively
    """
    candidates_result = list()
    candidates_probs = list()

    with open(input, 'r', encoding="utf-8") as fin:
        cloze = fin.read()

    sentences = map(lambda s: s.replace("\n", " "), nltk.tokenize.sent_tokenize(cloze))

    for sentence in sentences:
        sentence = ["<s>"] + sentence.lower().split() + ["</s>"]
        for i in range(1, len(sentence) - 1):
            if sentence[i] == EMPTY_SPACE:
                for candidate in candidates:
                    candidates_probs.append(get_prob(candidate, bigrams, trigrams, sentence, i, len(lexicon)))
                max_prob_index = np.argmax(candidates_probs)
                candidates_result.append(candidates[max_prob_index])
                candidates.pop(max_prob_index)
                candidates_probs.clear()

    return candidates_result


def solve_cloze(input, candidates, lexicon, corpus):

    print(f'starting to solve the cloze {input} with {candidates} using {lexicon} and {corpus}')

    candidates_list = read_file_to_list(candidates)
    # candidates_list_copy = copy.deepcopy(candidates_list)

    lexicon_set = set(read_file_to_list(lexicon))
    lexicon_set.update(["<s>", "</s>"])

    # Creating trigrams and bigrams data structures
    bigrams = create_bigrams(corpus, lexicon_set, candidates_list)
    trigrams = create_trigrams(corpus, lexicon_set, candidates_list)

    candidates_result = choose_candidates(input, candidates_list, bigrams, trigrams, lexicon_set)

    """
    print("Mean-random chance accuracy: ", compute_mean_random_chance_accuracy(candidates_list_copy))
    print("ML chance accuracy: ", compute_chance_accuracy(candidates_list_copy, candidates_result))
    """
    return candidates_result


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['lexicon_filename'],
                           config['corpus'])

    print('cloze solution:', solution)