# TO DO
# Need the 5 letter word list

# Create Prob Dist of a GUESS WORD
## Calculate count of words generating each pattern from word_list for the ANSWER_WORD -- DONE


# Given a word and a pattern choose all words that fit that pattern -- DONE: To shorten the word_list

# Given a word calculate entropy

# Compare guess word to actual word and output a pattern -- DONE


# To obtain pattern for the Guess Word for further proecssing
def getPattern(ans_word, guess):
    '''
    :param word: The guessed word
    :param ans_word: The correct answer
    :return: Pattern to assist next guess
        0 - Letter does not exist
        1 - Letter exits in the same location
        2 - Letter exists somewhere else
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
            pattern.append('1')
            letter_set[g_letter] -= 1
        elif g_letter in ans_word:
            if letter_set[g_letter] > 0:
                pattern.append('2')
                letter_set[g_letter] -= 1
            else:
                pattern.append('0')
        else:
            pattern.append('0')


    pattern = ''.join(pattern)
    return pattern


# To shorten word list based on pattern output
def searchPattern(ans_word, pattern, word_list):
    '''
    :param ans_word
    :param pattern: The specific pattern
    :param word_list: Set of all words in the list
    :return: Dict of all word that have the same pattern
    Given a GUESS WORD and a PATTERN find all possible words that could generate that pattern
    '''

    matchedWords = set()
    pattern1 = pattern
    pattern2 = ''.join(['1' if x == '2' else x for x in pattern1])

    for item in word_list:
        if getPattern(ans_word, item) == pattern1 or getPattern(ans_word, item) == pattern2:
            matchedWords.add(item)

    matchedWords.add(ans_word)
    return matchedWords


# Get the number of words generating each pattern for the given Answer
def patternDist(ans_word, word_list):
    '''
    :param word: Guessed word
    :param word_list: Dict of all words in the list
    :return: Dict of probability values for each pattern
    Given the ANSWER, calculate the count of each pattern
    '''

    prob_dist = {}

    for i in range(3):
        for j in range(3):
            for k in range(3):
                matched_words = set()
                pattern = str(i) + str(j) + str(k)
                for word in word_list:
                    if getPattern(ans_word, word) == pattern:
                        matched_words.add(word)
                prob_dist[pattern] = matched_words

    return prob_dist



# def wordProbDist(guess_word, word_list):




word_list = {'not', 'vet', 'bat', 'why', 'job', 'cat', 'lot', 'mad', 'cry', 'sit', 'one', 'hat', 'got', 'red',
             'cut', 'are', 'fog', 'two', 'zoo', 'fit', 'wry', 'ace', 'put'}

