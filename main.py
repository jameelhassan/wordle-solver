import numpy as np
import random
import itertools as it


EXACT = 2
SHIFTED = 1
NO_MATCH = 0

ALL_WORDS = 'data/allowed_words.txt'
ANS_WORDS = 'data/possible_words.txt'

PATTERN_MATRIX_FILE = 'pattern_matrix.npy'
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
        word_to_index = PATTERN_GRID['word_to_index']
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
        (w, int(w in ans_words) for w in all_words)
    )


def get_weights(word_list, priors):
    word_freqs = [priors[word] for word in word_list]
    total_freq = np.sum(word_freqs)
    if total_freq == 0:
        return np.zeros(word_freqs.shape)
    word_freqs /= total_freq
    return word_freqs


def get_pattern_dist(allowed_words, possible_words, weights):
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
    tot_possible_words = len(possible_words)
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)
    pattern_dist = np.zeros((tot_words, 3**5))
    for i, prob in weights:
        pattern_dist[np.arange(tot_words), pattern_matrix[:, i]] += prob
    return pattern_dist


def compute_entropy(prob_distr):
    entropy = 0
    for pattern, prob in prob_distr.items():
        entropy += -prob * np.log2(prob) if prob > 0 else 0
    return entropy


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



#
# word_list = {'clap', 'clip', 'luck', 'call', 'cute', 'lick', 'step', 'stop', 'site', 'grow', 'show',
#              'meow', 'plot', 'worm', 'stem', 'ease', 'stew', 'home', 'gone', 'moan', 'film', 'fine',
#              'bone', 'brim'}

if __name__ == "__main__":
    CORRECT = '22222'

    word_list = get_word_list()
    answer_words = get_word_list(ans_list=True)
    answer_word = random.choice(answer_words)
    print(f"There are {len(word_list)} possible words")
    attempt = 1

    max_entropy = 0
    words_traversed = 0
    for word in word_list:
        word_prob_distr = get_pattern_dist(word, word_list)
        word_entropy = compute_entropy(word_prob_distr)
        if word_entropy > max_entropy:
            max_entropy = word_entropy
            guess_word = word
        words_traversed += 1
        if words_traversed % 20 == 0:
            print('Words traversed {}'.format(words_traversed))

    pattern = get_pattern(answer_word, guess_word)
    print(f'Chosen guess word is {word} and the pattern is {pattern}')

    while pattern != CORRECT:
        word_list = filter_word_list(answer_word, pattern, word_list)
        print(f"Filtered list has {len(word_list)} words")

        max_entropy = 0
        for word in word_list:
            word_prob_distr = get_pattern_dist(word, word_list)
            word_entropy = compute_entropy(word_prob_distr)
            if word_entropy > max_entropy:
                max_entropy = word_entropy
                guess_word = word
        pattern = get_pattern(answer_word, guess_word)
        attempt += 1
        print(f'Chosen guess word is {word} and the pattern is {pattern}')

    print(f'''The answer was guessed in {attempt} attempts
The guessed answer is {guess_word}
The correct answer is {answer_word}
''')



