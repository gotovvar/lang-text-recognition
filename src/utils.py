import json
from typing import List, Tuple

PATH = "/home/user/lang-text-recognition/src/abstracting/corpus/texts_info.json"

def load_documents_and_languages(path: str = PATH) -> Tuple[List[str], List[str]]:
    """
    Loads document texts and their languages from a JSON file.

    Args:
        path (str): Path to JSON file containing information about documents. The PATH path is used by default.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - the first list contains the texts of the documents,
            - the second list contains the languages corresponding to each document.
    """   
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    languages = []
    
    for entry in data:
        text_file_path = entry['text_file']
        with open(text_file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
            languages.append(entry['language'])
    
    return documents, languages