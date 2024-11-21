from bs4 import BeautifulSoup
from .neural import LanguageClassifier
from .alphabet import recognize_language as alphabet_recognize_language
from .n_gram import recognize_language as n_gram_recognize_language
from fastapi import UploadFile
from typing import Tuple
from src.models.models import RecognitionMethod

#classifier = LanguageClassifier()

async def resolve(file: UploadFile, method: RecognitionMethod) -> Tuple[str, str]:
    """
    Specifies the language of the text extracted from the HTML file using the specified recognition method.

    Args:
        file (UploadFile): Uploaded HTML file containing the text to be analyzed.
        method (RecognitionMethod): The method for recognizing the language of the text.
                                    Possible values:
                                    - RecognitionMethod.NGRAM
                                    - RecognitionMethod.ALPHABET
                                    - RecognitionMethod.NEURAL

    Returns:
        Tuple[str, str]: Tuple containing:
                         - The language of the text (e.g., 'russian' or 'italian').
                         - Extracted text from the HTML file.
    """
    content = await file.read()
    content_str = content.decode("utf-8")

    soup = BeautifulSoup(content_str, "html.parser")
    extracted_text = soup.get_text(separator=" ", strip=True)

    if method == RecognitionMethod.NGRAM:
        language = n_gram_recognize_language(extracted_text)
    elif method == RecognitionMethod.ALPHABET:
        language = alphabet_recognize_language(extracted_text)
    #elif method == RecognitionMethod.NEURAL:
        #language = classifier.predict_language(extracted_text)

    return language, extracted_text