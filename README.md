# wordle-solver
A wordle solver using an Information Theoretic approach

A complete implementation from scratch inspired from the work of [3Blue1Brown](https://www.youtube.com/watch?v=v68zYyaEmEA&t=1049s). 

## Approach

The idea is to make guesses maximizing the Expected Information (Entropy) from the distribution of words. At each stage, the Entropy of each word is computed and the word having the maximum entropy is chosen as the guess.

![Sample game](https://github.com/jameelhassan/wordle-solver/blob/main/results/wordle_pattern.png?raw=true)

