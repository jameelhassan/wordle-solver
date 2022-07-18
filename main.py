import numpy as np
import random

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
        if prob != 0:
            entropy += -prob * np.log(prob)

    return entropy

#
# word_list = {'clap', 'clip', 'luck', 'call', 'cute', 'lick', 'step', 'stop', 'site', 'grow', 'show',
#              'meow', 'plot', 'worm', 'stem', 'ease', 'stew', 'home', 'gone', 'moan', 'film', 'fine',
#              'bone', 'brim'}

if __name__ == "__main__":
    ALL_WORDS = 'data/allowed_words.txt'
    ANS_WORDS = 'data/possible_words.txt'
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



