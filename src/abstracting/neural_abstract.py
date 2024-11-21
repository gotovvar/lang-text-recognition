import nltk
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

class BilingualSummarizer:
    """
    A class for summarizing text in two languages, Russian and Italian, using the T5 and Pegasus models.

    This class uses pre-trained models for text summarization: 
    - For Russian, the T5 model (UrukHan/t5-russian-summarization) is used.
    - For Italian, the Pegasus model is used (google/pegasus-xsum).

    The main goal of the class is to divide the text into parts that meet the length constraints of the model,
    and then summarize these parts, creating a final summary.

    Attributes:
        max_length (int): The maximum length of the final summarized text (default is 150).
        min_length (int): Minimum length of the final summarized text (default is 10).

    Methods:
        __init__(max_length: int = 150, min_length: int = 10) -> None: Initializes the class with the specified maximum and minimum length of the summarize text.
        summarize_text(text: str, language: str) -> str: Performs summarization of the text for the specified language, splitting it into parts and summarizing each part.
        split_text_into_parts(sentences: List[str], tokenizer, language: str, max_length: int = 300) -> List[str]: Splits text into parts depending on sentence length and model constraints.
        summarize_part(part: str, model, tokenizer) -> str: Summarizes one part of text using the specified model and tokenizer.
    """
    def __init__(self, max_length: int = 150, min_length: int = 10):
        self.models = {
            "russian": {
                "tokenizer": T5Tokenizer.from_pretrained("UrukHan/t5-russian-summarization"),
                "model": T5ForConditionalGeneration.from_pretrained("UrukHan/t5-russian-summarization")
            },
            "italian": {
                "tokenizer": PegasusTokenizer.from_pretrained("google/pegasus-xsum"),
                "model": PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
            }
        }
        self.max_length = max_length
        self.min_length = min_length

    def summarize_text(self, text: str, language: str) -> str:
        """
        Summarizes the text based on the selected language. The text is broken into parts and each part is summarized separately.
        The summarized parts are then combined into a final result.

        Args:
            text (str): Text to be summarized.
            language (str): The language of the text should be either “russian” or “italian”.

        Returns:
            str: Final summary of the text.
        """
        if language not in self.models:
            raise ValueError(f"Language '{language}' is not supported. Supported languages are: {', '.join(self.models.keys())}")

        tokenizer = self.models[language]["tokenizer"]
        model = self.models[language]["model"]

        sentences = nltk.sent_tokenize(text)
        parts = self.split_text_into_parts(sentences, tokenizer, language)

        summaries = [
            self.summarize_part(part, model, tokenizer)
            for part in tqdm(parts, desc=f"Summarizing text in {language}")
        ]
        
        final_summary = " ".join(summaries)
        return final_summary

    def split_text_into_parts(self, sentences: list, tokenizer, language: str, max_length: int = 300) -> list:
        """
        Breaks the text into parts. For Russian - into pairs of sentences.
        For other languages, into parts, each not exceeding max_length of tokens.

        Args:
            sentences (list): A list of sentences to be broken down into parts.
            tokenizer: Tokenizer used to convert text into tokens.
            language (str): The language of the text that defines the logic of the partitioning.
            max_length (int): Maximum length of one part (default is 300).

        Returns:
            list: A list of text parts, each of which does not exceed max_length tokens.
        """
        if language == "russian":
            parts = [sentence for sentence in sentences if len(sentence) >= 200]
            return parts
        
        else:
            current_part = []
            current_length = 0
            parts = []
            
            for sentence in sentences:
                sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                sentence_length = len(sentence_tokens)
                
                if current_length + sentence_length <= max_length:
                    current_part.append(sentence)
                    current_length += sentence_length
                else:
                    parts.append(" ".join(current_part))
                    current_part = [sentence]
                    current_length = sentence_length
            
            if current_part:
                parts.append(" ".join(current_part))
            
            return parts

    def summarize_part(self, part: str, model, tokenizer) -> str:
        """
        Summarizes one piece of text using the specified model and tokenizer.

        Args:
            part (str): The part of the text to be summarized.
            model: A model for text summarization.
            tokenizer: Tokenizer to convert text into the desired format for the model.

        Returns:
            str: Summarized sentence.
        """
        inputs = tokenizer(part, return_tensors="pt", truncation=True, padding="longest", max_length=512)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            min_length=self.min_length,
            length_penalty=1.0,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
