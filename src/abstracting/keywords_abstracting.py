import yake
from nltk.corpus import stopwords
from nltk import word_tokenize



def extract_keywords(text: str, language: str) -> str:
    """
    Extracts keywords from text using the YAKE library.

    Args:
        text (str): The text from which you need to extract keywords.
        language (str): The language of the text (e.g. 'ru' for Russian, 'en' for English).
                        The language must be specified in a format supported by the YAKE library.

    Returns:
        str: A string containing keywords separated by commas.
    """
    preprocessed_text = preprocess_text(text, language)

    custom_kw_extractor = yake.KeywordExtractor(lan=language, 
                                                        n=2, 
                                                        dedupLim=0.3, 
                                                        dedupFunc='seqm', 
                                                        windowsSize=1, 
                                                        top=10, 
                                                        features=None)
    
    keywords = custom_kw_extractor.extract_keywords(preprocessed_text)

    top_keywords = [kw[0] for kw in keywords]
    return ', '.join(top_keywords)


def preprocess_text(text: str, language: str) -> str:
    """
    Pre-processes text for further analysis by removing stop words and characters other than letters.

    Args:
        text (str): Text to be preprocessed.
        language (str): Text language to load the appropriate stop words from NLTK.

    Returns:
        str: Pre-processed text consisting of words separated by spaces.
    """
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words(language))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    return ' '.join(words)

 