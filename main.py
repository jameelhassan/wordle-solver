# TO DO
# Need the 5 letter word list

# Create Prob Dist of a GUESS WORD
## Calculate count of words generating each pattern from word_list for the ANSWER_WORD -- DONE


# Given a word and a pattern choose all words that fit that pattern -- DONE: To shorten the word_list

# Given a word calculate entropy

# Compare guess word to actual word and output a pattern -- DONE


# To obtain pattern for the Guess Word for further processing
def getPattern(ans_word, guess):
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


# To shorten word list based on pattern output
def filterWords(ans_word, pattern, word_list):
    '''
    :param ans_word
    :param pattern: The specific pattern
    :param word_list: Set of all words in the list
    :return: Set of all word that have the same pattern
    Given a GUESS WORD and a PATTERN find all possible words that could generate that pattern
    '''

    matchedWords = set()

    for item in word_list:
        # Set each word as the answer, and check the pattern if the ans_word is used
        if getPattern(item, ans_word) == pattern:
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
                for l in range(3):
                    matched_words = 0
                    pattern = str(i) + str(j) + str(k) + str(l)
                    for word in word_list:
                        if getPattern(ans_word, word) == pattern:
                            matched_words += 1

                    prob_dist[pattern] = matched_words

    return prob_dist



# def wordProbDist(guess_word, word_list):




word_list = {'clap', 'clip', 'luck', 'call', 'cute', 'lick', 'step', 'stop', 'site', 'grow', 'show',
             'meow', 'plot', 'worm', 'stem', 'ease', 'stew', 'home', 'gone', 'moan', 'film', 'fine',
             'bone', 'brim'}

if __name__ == "__main__":
    answer = 'luck'
    guess = 'call'

    pattern = getPattern(answer, guess)
    pattern_match = filterWords(answer, pattern, word_list)
    distr = patternDist(answer, word_list)

    print(pattern)
    print("words that matched the pattern are")
    for w in pattern_match:
        print(w)

    print(distr)

    total = 0
    for count in distr.values():
        total += count

    print(f"There are {len(word_list)} number of words and {total} in distr")