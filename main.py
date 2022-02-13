# TO DO
# Need the 5 letter word list

# Calculate probability of a pattern for a given word and pattern
## Given a word and a pattern choose all words that fit that pattern

# Given a word calculate entropy

# Compare guess word to actual word and output a pattern -- DONE



def getPattern(word, ans_word):
    '''
    :param word: The guessed word
    :param ans_word: The correct answer
    :return: Pattern to assist next guess
        0 - Letter does not exist
        1 - Letter exits in the same location
        2 - Letter exists somewhere else
    '''

    pattern = []
    word_ = list(word)
    ans_word_ = list(ans_word)

    for idx, letter in enumerate(word_):
        if ans_word_[idx] == letter:
            pattern.append('1')
        else:
            if letter in ans_word_:
                pattern.append('2')
            else:
                pattern.append('0')

    pattern = ''.join(pattern)
    return pattern


def searchPattern(word, pattern, word_list):
    '''
    :param pattern: The specific pattern
    :param word_list: Set of all words in the list
    :return: Dict of all word that have the same pattern
    '''

    matchedWords = set()
    pattern1 = pattern
    pattern2 = ''.join(['1' if x == '2' else x for x in pattern1])

    for item in word_list:
        if getPattern(word, item) == pattern1 or getPattern(word, item) == pattern2:
            matchedWords.add(item)

    matchedWords.add(word)
    return matchedWords




word_list = {'not', 'vet', 'bat', 'why', 'job', 'cat', 'lot', 'mad', 'cry', 'sit', 'one', 'hat', 'got', 'red',
             'cut', 'are', 'fog', 'two', 'zoo', 'fit', 'wry', 'ace', 'put'}

