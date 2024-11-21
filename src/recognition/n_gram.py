import re
import json
import math
from collections import Counter
from typing import Mapping , List


def preprocess_text(text: str) -> str:
    """
    Prepares text by removing all characters except letters.
    
    Args:
        text (str): Input text for analysis.
    
    Returns:
        str: The processed text consists only of letters in the saved registry.
    """
    text = re.sub(r'[^a-zа-яёàèìòù]', '', text.lower())
    return text

def create_ngrams(text: str, n: int) -> List[str]:
    """
    Generates n-grams from processed text.

    Args:
        text (str): Text for generating n-grams.
        n (int): Dimension n-grams.

    Returns:
        List[str]: List of n-grams extracted from text.
    """
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return ngrams

def build_profile(text: str, n: int = 2) -> Mapping[str, int]:
    """
    Creates a text profile as an n-gram frequency distribution.

    Args:
        text (str): Text for building a profile.
        n (int): Dimension n-grams (default - 2).

    Returns:
        Mapping[str, int]: Text profile, which is a dictionary of n-grams and their frequencies.
    """
    processed_text = preprocess_text(text)
    ngrams_generated = create_ngrams(processed_text, n)
    ngram_freq = Counter(ngrams_generated)
    total_ngrams = sum(ngram_freq.values())
    profile = {''.join(ngram): freq / total_ngrams for ngram, freq in ngram_freq.items()}
    return profile

def calculate_kullback_leibler_distance(user_profile: Mapping[str, int], language_profile: Mapping[str, int]) -> float:
    """
    Calculates the Kullback-Leibler distance between the user's text profile and the language profile.

    Args:
        user_profile (Mapping[str, int]): User profile with n-gram frequencies.
        language_profile (Mapping[str, int]): Language profile for comparison.

    Returns:
        float: Kullback-Leibler distance between two profiles.
    """
    distance = 0.0
    for ngram in user_profile:
        p_input = user_profile.get(ngram, 0.0)
        p_lang = language_profile.get(ngram, 1e-10)
        distance += p_input * math.log(p_input / p_lang)
    return distance

def recognize_language(text: str, n: int = 2) -> str:
    """
    Determines the language of the text by comparing the text profile with language profiles based on n-grams and Kullback-Leibler distance.

    Args:
        text (str): Text for language recognition.
        n (int): Dimension n-grams (default - 2).

    Returns:
        str: The name of the language that most likely matches the text.
    """
    with open('src/recognition/datasets_profile/italian_language_profile.json') as file:
        it_dataset_profile = json.load(file)
            
    with open('src/recognition/datasets_profile/russian_language_profile.json') as file:
        ru_dataset_profile = json.load(file)

    language_profiles = {
        'italian': it_dataset_profile,
        'russian': ru_dataset_profile,
    }

    user_profile = build_profile(text, n)
    distances = {}
    for language, language_profile in language_profiles.items():
        distances[language] = calculate_kullback_leibler_distance(user_profile, language_profile)
    return min(distances, key=distances.get)