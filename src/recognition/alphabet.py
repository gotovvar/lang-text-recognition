from collections import Counter
from typing import Mapping
import re

alphabet_frequencies = {
    'russian': {
        "а": 0.0817,
        "б": 0.0159,
        "в": 0.0453,
        "г": 0.0170,
        "д": 0.0356,
        "е": 0.0843,
        "ё": 0.0020,
        "ж": 0.0054,
        "з": 0.0135,
        "и": 0.0709,
        "й": 0.0150,
        "к": 0.0350,
        "л": 0.0426,
        "м": 0.0294,
        "н": 0.0670,
        "о": 0.1095,
        "п": 0.0271,
        "р": 0.0422,
        "с": 0.0544,
        "т": 0.0657,
        "у": 0.0231,
        "ф": 0.0022,
        "х": 0.0060,
        "ц": 0.0047,
        "ч": 0.0156,
        "ш": 0.0069,
        "щ": 0.0016,
        "ъ": 0.0007,
        "ы": 0.0193,
        "ь": 0.0185,
        "э": 0.0123,
        "ю": 0.0077,
        "я": 0.0202,
    },
    'italian': {
        "e": 0.1177,
        "a": 0.1172,
        "i": 0.1012,
        "o": 0.0983,
        "n": 0.0688,
        "r": 0.0635,
        "t": 0.0561,
        "l": 0.0560,
        "s": 0.0496,
        "c": 0.0450,
        "d": 0.0373,
        "p": 0.0305,
        "u": 0.0301,
        "m": 0.0251,
        "g": 0.0164,
        "v": 0.0163,
        "b": 0.0092,
        "z": 0.0049,
        "f": 0.0095,
        "q": 0.0051,
        "h": 0.0049,
        "x": 0.0000,
        "j": 0.0000,
        "k": 0.0000,
        "w": 0.0000,
        "y": 0.0000,
        "à": 0.0110,
        "è": 0.0105,
        "é": 0.0019,
        "ì": 0.0053,
        "ò": 0.0065,
        "ù": 0.0021 
  }
}

def preprocess_text(text: str) -> Counter:
    """
    Prepares text by removing all characters except letters and counts the frequency of each character.
    
    Args:
        text (str): Input text for analysis.
    
    Returns:
        Counter: A counter with the number of each character in the text.
    """
    
    text = re.sub(r'[^a-zа-яёàèìòù]', '', text.lower())
    return Counter(text)


def build_profile(text: str) -> Mapping[str, int]:
    """
    Creates a character frequency profile for text based on relative frequencies.
    
    Args:
        text (str): Input text for analysis.
    
    Returns:
        Mapping[str, int]: Symbol frequency profile, where keys are symbols and values ​​are their frequency.
    """
    letter_counts = preprocess_text(text)
    total_letters = sum(letter_counts.values())
    profile = {char: count / total_letters for char, count in letter_counts.items()}
    return profile


def calculate_manhattan_distance(user_profile: Mapping[str, int], language_profile: Mapping[str, int]) -> float:
    """
    Calculates the Manhattan distance between the text profile and the language profile.
    
    Args:
        user_profile (Mapping[str, int]): Text character profile for the user.
        language_profile (Mapping[str, int]): Character frequency profile for a language.
    
    Returns:
        float: Manhattan distance between text profile and language profile.
    """
    all_keys = set(user_profile.keys()).union(language_profile.keys())
    distance = 0
    for key in all_keys:
        distance += abs(user_profile.get(key, 0) - language_profile.get(key, 0))
    return distance


def recognize_language(text: str) -> str:
    """
    Determines the likely language of the text based on the character frequency profile and Manhattan distance.
    
    Args:
        text (str): Input text for analysis.
    
    Returns:
        str: Intended language of the text.
    """
    user_profile = build_profile(text)
    
    distances = {}
    for language, language_profile in alphabet_frequencies.items():
        distance = calculate_manhattan_distance(user_profile, language_profile)
        distances[language] = distance
    
    predicted_language = min(distances, key=distances.get)
    return predicted_language
