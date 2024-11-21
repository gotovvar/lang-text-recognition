import numpy as np
from collections import Counter
from typing import List
from math import log
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class TextSummarizer:
    """
    The TextSummarizer class is designed to create summarizations of text based on the TF-IDF model, taking into account the position of the sentences.
    This class takes text data and highlights key sentences to create summarizations.

    Attributes:
        documents (List[str]): A list of documents used to calculate frequency statistics for summarization.
        languages (List[str]): A list of languages corresponding to each document, used for proper tokenization and stopword removal.
        doc_count (int): The total number of documents in the corpus, used for calculating IDF (Inverse Document Frequency).
        df (Counter): A frequency dictionary of terms across all documents, used for calculating TF-IDF.

    Methods:
        __init__(documents: List[str], languages: List[str]) -> None: Initializes the class with a set of documents and their languages.
        _calculate_document_frequency() -> Counter: Calculates the document frequency for terms across the loaded documents.
        _preprocess_text(text: str, language: str) -> List[str]: Preprocesses text by tokenizing and removing stopwords.
        _calculate_tf_idf(sentence: str, document: str, language: str) -> float: Calculates the TF-IDF score for a given sentence.
        _calculate_position_scores(sentences: List[str], document: str) -> List[float]: Calculates position-based scores for each sentence.
        summarize(document: str, language: str, num_sentences=10) -> str: Summarizes the given document using the previously loaded data for TF-IDF and position scoring.
    """
    def __init__(self, documents: List[str], languages: List[str]):
        """
        Initializes the class with a set of documents and their languages.
        """
        self.documents = documents
        self.languages = languages
        self.doc_count = len(documents)
        self.df = self._calculate_document_frequency()

    def _calculate_document_frequency(self) -> Counter:
        """
        Counts the number of documents that contain a word from each document

        Returns:
            Counter: A dictionary with frequencies of words occurring in documents.
        """
        df = Counter()
        for doc in self.documents:
            language = self.languages[self.documents.index(doc)]
            words = set(self._preprocess_text(doc, language))
            df.update(words)
        return df

    def _preprocess_text(self, text: str, language: str) -> List[str]:
        """
        Pre-processes text: tokenizes, removes stop words and leaves only alphabetic words.

        Args:
            text (str): Pre-processing text.
            language (str): Text language (used for proper tokenization and stop words).

        Returns:
            List[str]: A list of words from the text after preprocessing.
        """
        words = word_tokenize(text, language=language)
        stop_words = set(stopwords.words(language))
        words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
        return words

    def _calculate_tf_idf(self, sentence: str, document: str, language: str) -> float:
        """
        Calculates the TF-IDF value for a sentence in the document.

        Args:
            sentence (str): The proposal for which the TF-IDF is calculated.
            document (str): The document to which the proposal belongs.
            language (str): Text language (required for correct preprocessing).

        Returns:
            float: TF-IDF value for the proposal.
        """
        words = self._preprocess_text(sentence, language)
        tf = Counter(words)
        tfmax = max(tf.values())
        score = 0

        for term, freq in tf.items():
            tf_t_si = freq / len(words)
            tf_t_d = document.count(term)
            w_t_d = 0.5 * (1 + tf_t_d / tfmax) * log(self.doc_count / (1 + self.df[term]))
            score += tf_t_si * w_t_d
        return score

    def _calculate_position_scores(self, sentences: str, document: str) ->  List[float]:
        """
        Calculates positional scores for sentences in a document based on their positioning.

        Args:
            sentences (str): A list of the sentences that make up the document.
            document (str): Document text.

        Returns:
            List[float]: A list of positional points for each sentence.
        """
        position_scores = []
        total_chars = len(document)
        
        for i, sentence in enumerate(sentences):
            chars_before_sent = sum(len(sentences[j]) for j in range(i))
            chars_in_paragraph = len(sentence)
            
            posd_si = 1 - (chars_before_sent / total_chars)

            posp_si = 1 - (chars_before_sent / chars_in_paragraph) if chars_in_paragraph > 0 else 0
            position_scores.append(posd_si * posp_si)
        
        return position_scores

    def summarize(self, document: str, language: str, num_sentences=10) -> str:
        """
        Selects the most relevant proposals based on TF-IDF and position scores.

        Args:
            document (str): Document text for summarization.
            language (str): Document Language.
            num_sentences (int): Number of proposals to be included in the summarization.

        Returns:
            str: Key sentences.
        """        
        sentences = sent_tokenize(document, language=language)
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            tf_idf_score = self._calculate_tf_idf(sentence, document, language)
            position_score = self._calculate_position_scores(sentences, document)[i]
            sentence_scores.append(tf_idf_score * position_score)
        
        top_sentences = np.argsort(sentence_scores)[-num_sentences:]
        top_sentences = sorted(top_sentences)
        
        summary = " ".join([sentences[i] for i in top_sentences])
        return summary
