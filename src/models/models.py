from enum import Enum

class RecognitionMethod(Enum):
    NGRAM = 'ngram'
    ALPHABET = 'alphabet'
    NEURAL = 'neural'

    