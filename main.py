import numpy as np
import random
import itertools as it
from scipy.stats import entropy
import os
import json


EXACT = 2
SHIFTED = 1
NO_MATCH = 0

ALL_WORDS = 'data/allowed_words.txt'
ANS_WORDS = 'data/possible_words.txt'
WORD_FREQ = 'data/word_freq.json'
PATTERN_MATRIX_FILE = 'data/pattern_matrix.npy'
PATTERN_GRID = dict()


def get_word_list(ans_list=False):
    '''
    :param filename: path to txt file with words
    :return: list of all words
    '''
    filename = ANS_WORDS if ans_list else ALL_WORDS
    with open(filename, 'r') as f:
        word_list = f.read().split('\n')
    return word_list


def get_pattern(guess_word, ans_word):
    '''
    :param word: The guessed word
    :param ans_word: The correct answer
    :param PATTERN_GRID: The complete pattern matrix and word_to_index mapping stored in a dictionary
    :return: Pattern representing wordle similarity
        0 - Letter does not exist
        2 - Letter exits in the same location
        1 - Letter exists somewhere else
    Given a GUESS WORD and ANSWER find the pattern it would generate
    '''

    if PATTERN_GRID:
        word_to_index = PATTERN_GRID['words_to_index']
        if ans_word in word_to_index and guess_word in word_to_index:
            pattern = get_pattern_matrix([guess_word], [ans_word])[0,0]
    pattern = generate_pattern_matrix([guess_word], [ans_word])[0,0]
    return pattern


def filter_word_list(guess_word, pattern, word_list):
    guess_patterns = get_pattern_matrix([guess_word], word_list)
    filtered_word_list = np.array(word_list)[guess_patterns.flatten() == pattern]
    return filtered_word_list


def get_wordle_prior():
    all_words = get_word_list()
    ans_words = get_word_list(ans_list=True)
    return dict(
        (w, int(w in ans_words)) for w in all_words
    )


def get_weights(word_list, priors):
    word_freqs = np.array([priors[word] for word in word_list], dtype=np.float)
    total_freq = np.sum(word_freqs)
    if total_freq == 0:
        return np.zeros(word_freqs.shape)
    word_freqs /= total_freq
    return word_freqs


def get_word_freq():
    if os.path.exists(WORD_FREQ):
        with open(WORD_FREQ, 'r') as js:
            word_freqs = json.load(js)
        return word_freqs
    raise Exception(f"Word frequency file {WORD_FREQ} not found")


def get_freq_priors(num_of_common_words=3000, sigmoid_width=10):
    '''
    Using the frequency(as a probability) of each 5-letter word, a new weight is given
    for each word. Since two words with significantly different frequencies can have
    equally likely chance of being a wordle answer, a binary-like output weight using
    a sigmoid is generated.

    This can be conceptualized as taking the word list sorted by frequencies and considering
    the indexes as a random variable. This is then transformed to a new random variable
    such that the first of the common words (n_common_words from reverse) falls to zero.
    Transformation Y = kX + delta.
    '''

    word_frequencies = get_word_freq()
    words = np.array([w for w in word_frequencies.keys()])
    frequencies = np.array([word_frequencies[w] for w in words])
    argsort_ids = np.argsort(frequencies)
    sorted_words = words[argsort_ids]

    center = sigmoid_width * (num_of_common_words / len(words))
    ys = np.linspace(center - sigmoid_width / 2, center + sigmoid_width / 2, len(words))

    word_priors = dict()
    for word, y in zip(sorted_words, ys):
        word_priors[word] = sigmoid(y)
    return word_priors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_pattern_distribution(allowed_words, possible_words, weights=None):
    '''
    For each allowed guess word, there are 3**5 possible patterns that could occur. These patterns will depend
    on the possible answer words and their corresponding probability of occuring

    The function return the probability distribution of the 3**5 patterns for each of the allowed guess word by
    evaluating the pattern and aggregating the probability of each possible answer word in weights accordingly.
    :param allowed_words:
    :param possible_words:
    :param weights:
    :return:
    '''
    tot_words = len(allowed_words)
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)
    pattern_dist = np.zeros((tot_words, 3**5))
    weights = 1 / tot_words * np.ones(tot_words, dtype=np.float) if weights is None else weights
    for i, prob in enumerate(weights):
        pattern_dist[np.arange(tot_words), pattern_matrix[:, i]] += prob
    return pattern_dist


def compute_entropy(pattern_dist):
    ax = len(pattern_dist.shape) - 1
    return entropy(pattern_dist, base=2, axis=ax)


def maximize_entropy(entropies, n_top=None):
    if n_top is None:
        best_idx = [np.argmax(entropies)]
    else:
        best_idx = np.argsort(entropies)[::-1][:n_top]
    return best_idx


def convert_to_ascii(word_list):
    word_arr = np.array([[ord(w) for w in word] for word in word_list])
    return word_arr


def generate_pattern_matrix(words1, words2):
    '''
    A pattern for each pair of words in the two word lists in the sense of wordle-similarity is generated.
    The pattern is of the form 0 --> Grey, 1 --> Yellow, 2 --> Green.

    The function generates the pattern for each pair of words and returns an numpy array of shape
    (#words1, #words2, word_length)

    :param words1: word list 1
    :param words2: word list 2
    :return: A numpy array of the patten matrix. For each Guess, what would be the pattern for
    each answer (dimension order)
    '''

    nw1 = len(words1)
    nw2 = len(words2)
    nl = len(words1[0])

    words_arr1, words_arr2 = map(convert_to_ascii, (words1, words2))

    equality_matrix = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    # equality matrix holds True if words[a][i] = words[b][i]
    # This shows which letters of which words are the same in a grid
    for i in range(nl):
        for j in range(nl):
            equality_matrix[:, :, i, j] = np.equal.outer(words_arr1[:, i], words_arr2[:, j])

    pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)
    # Set EXACT matches
    for i in range(nl):
        letter_matches = equality_matrix[:, :, i, i].flatten()
        pattern_matrix[:, :, i].flat[letter_matches] = EXACT

        for j in range(nl):
            # If a matched letter, is matching to a different letter as well, set
            # this to False to prevent a double trigger in the SHIFTED match pass
            equality_matrix[:, :, j, i].flat[letter_matches] = False
            equality_matrix[:, :, i, j].flat[letter_matches] = False

    # Set SHIFTED matches
    for i, j in it.product(range(nl), range(nl)):
        letter_matches = equality_matrix[:, :, i, j].flatten()
        pattern_matrix[:, :, i].flat[letter_matches] = SHIFTED

        for k in range(nl):
            # Similar to above, mark as taken care of
            equality_matrix[:, :, k, j].flat[letter_matches] = False
            equality_matrix[:, :, i, k].flat[letter_matches] = False

    # Storing the pattern as a ternary value simplifies in indexing using
    # the pattern for filtering the word list
    full_pattern_matrix = np.dot(pattern_matrix, 3**np.arange(nl)).astype(np.uint8)
    return full_pattern_matrix


def ternary_to_int_pattern(pattern):
    result = []
    value = pattern
    for i in range(5):
        result.append(value % 3)
        value = value // 3
    return result


def generate_full_pattern_matrix():
    all_words = get_word_list()
    full_pattern_matrix = generate_pattern_matrix(all_words, all_words)
    np.save(PATTERN_MATRIX_FILE, full_pattern_matrix)
    return full_pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID:
        if not PATTERN_MATRIX_FILE:
            print('''
            Need to generate the pattern matrix file. This will be stored to disk
            ''')
            generate_full_pattern_matrix()
        PATTERN_GRID['grid'] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID['words_to_index'] = dict(zip(get_word_list(), it.count()))

    full_pattern_matrix = PATTERN_GRID['grid']
    words_to_index = PATTERN_GRID['words_to_index']

    indices1 = [words_to_index[word] for word in words1]
    indices2 = [words_to_index[word] for word in words2]
    return full_pattern_matrix[np.ix_(indices1, indices2)]


def pattern_to_wordle_like(pattern):
    d = {NO_MATCH: "⬛", SHIFTED: "🟨", EXACT: "🟩"}
    return "".join(d[x] for x in ternary_to_int_pattern(pattern))


def gameplay(autoplay=True, priors=None):
    allowed_words = get_word_list()
    possible_words = get_word_list(ans_list=True)
    answer_word = np.random.choice(possible_words)
    guess = 'xxxxx'
    iters = 0

    while guess != answer_word:
        if priors is None:
            priors = get_wordle_prior()
        weights = get_weights(possible_words, priors)
        pattern_distribution = get_pattern_distribution(allowed_words, allowed_words)
        entropies = compute_entropy(pattern_distribution)
        best_guesses = maximize_entropy(entropies, n_top=6)

        # Take user input
        if not autoplay:
            word_picks = [allowed_words[idx] for idx in best_guesses]
            top_picks = dict(zip(word_picks, zip(best_guesses, entropies[best_guesses])))

            # Show the top word picks and E[I]
            for word, (idx, ent) in top_picks.items():
                print(f"{word.upper()} \t E[I]: {ent:.2f}")

            word_choice = input("Enter your choice of word\n").lower()
            while word_choice not in top_picks.keys():
                word_choice = input("Invalid choice.\nPlease choose from filtered list\n").lower()
                for word, (idx, ent) in top_picks.items():
                    print(f"{word.upper()} \t E[I]: {ent:.2f}")

            best_idx = top_picks[word_choice][0]
            guess = word_choice
        else:
            best_idx = np.random.choice(best_guesses)
            guess = allowed_words[best_idx]

        iters += 1
        pattern = get_pattern(guess, answer_word)
        guess_pattern_prob = pattern_distribution[best_idx, pattern]
        actual_info = np.log2(1 / guess_pattern_prob)
        print(f"E[I]: {entropies[best_idx]:.2f}, Actual Info: {actual_info:.2f}") if autoplay else None
        print(f"The guessed word is {guess.upper()} and the pattern is {pattern_to_wordle_like(pattern)}\n")

        allowed_words = filter_word_list(guess, pattern, allowed_words)
        possible_words = filter_word_list(guess, pattern, possible_words)

    print(f"The correct word is {answer_word.upper()}, and was guessed in {iters} attempts")


if __name__ == "__main__":
    gameplay()
    # print(game_result)



