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


def get_word_list(filename):
    '''
    :param filename: path to txt file with words
    :return: list of all words
    '''
    with open(filename, 'r') as f:
        word_list = f.read().split('\n')

    return word_list


def get_pattern(ans_word, guess):
    '''
    :param word: The guessed word
    :param ans_word: The correct answer
    :return: Pattern to assist next guess
        0 - Letter does not exist
        2 - Letter exits in the same location
        1 - Letter exists somewhere else
    Given a GUESS WORD and ANSWER find the pattern it would generate
    '''

    pattern = []
    letter_set = {}

    for letter in ans_word:
        if letter in letter_set:
            letter_set[letter] += 1
        else:
            letter_set[letter] = 1

    for idx, g_letter in enumerate(guess):
        if g_letter == ans_word[idx]:
            # Letter in the same location
            pattern.append('2')
            letter_set[g_letter] -= 1
        elif g_letter in ans_word:
            if letter_set[g_letter] > 0:
                # Letter exists in different location
                pattern.append('1')
                letter_set[g_letter] -= 1
            else:
                # Letter does not exist
                pattern.append('0')
        else:
            pattern.append('0')


    pattern = ''.join(pattern)
    return pattern


def filter_word_list(ans_word, pattern, word_list):
    '''
    :param ans_word
    :param pattern: The specific pattern
    :param word_list: Set of all words in the list
    :return: Set of all word that have the same pattern
    Filter the word list based on the pattern
    Given a WORD and a PATTERN find all possible words that could generate that pattern
    '''

    matchedWords = set()

    for item in word_list:
        if get_pattern(item, ans_word) == pattern:
            matchedWords.add(item)

    matchedWords.add(ans_word)
    return matchedWords


def get_pattern_dist(test_word, word_list):
    '''
    :param word: Guessed word
    :param word_list: Dict of all words in the list
    :return: Dict of probability values for each pattern
    Given a word, what is the probability distribution of all possible patterns
    '''

    prob_dist = {}
    total_words = len(word_list)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        matched_words = 0
                        pattern = str(i) + str(j) + str(k) + str(l) + str(m)
                        for word in word_list:
                            if get_pattern(test_word, word) == pattern:
                                matched_words += 1

                        prob_dist[pattern] = matched_words/total_words

    # score_sum = sum(prob_dist.values())
    # for pattern, score in prob_dist.items():
    #     prob_dist[pattern] = score/score_sum
    return prob_dist


def compute_entropy(prob_distr):
    entropy = 0
    for pattern, prob in prob_distr.items():
        entropy += -prob * np.log(prob) if prob > 0 else 0
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
    :return: A numpy array of the patten matrix
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
        pattern_matrix.flat[letter_matches] = EXACT

        for j in range(nl):
            # If a matched letter, is matching to a different letter as well, set
            # this to False to prevent a double trigger in the SHIFTED match pass
            equality_matrix[:, :, j, i].flat[letter_matches] = False
            equality_matrix[:, :, i, j].flat[letter_matches] = False

    # Set SHIFTED matches
    for i, j in it.product(range(nl), range(nl)):
        letter_matches = equality_matrix[:, :, i, j].flatten()
        pattern_matrix.flat[letter_matches] = SHIFTED

        for k in range(nl):
            # Similar to above, mark as taken care of
            equality_matrix[:, :, k, j].flat[letter_matches] = False
            equality_matrix[:, :, i, k].flat[letter_matches] = False

    return pattern_matrix


def generate_full_pattern_matrix():
    all_words = get_word_list(ALL_WORDS)
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
        PATTERN_GRID['words_to_index'] = dict(zip(words1, it.count()))

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

    word_list = get_word_list(ANS_WORDS)
    answer_words = get_word_list(ANS_WORDS)
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



