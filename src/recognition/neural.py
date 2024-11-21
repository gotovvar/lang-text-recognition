import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

PATH = '/home/user/lang-text-recognition/src/recognition/datasets_profile/Filtered_Language_Detection.csv'

class LanguageClassifier:
    """
    The LanguageClassifier class is designed to classify the language of a text based on a trained neural network model.
    This class accepts text data and determines in which language the text is written (Russian or Italian).

    Attributes:
        data (pd.DataFrame): Data with texts and language labels.
        vectorizer (CountVectorizer): Text vectorizer for converting text data into numeric form.
        model (Sequential): A neural network for text language classification.

    Methods:
        __init__(): Loads and prepares data and trains a model on textual data.
        predict_language(text: str) -> str: Accepts a string of text and returns the predicted language ('russian' or 'italian').
    """
    def __init__(self):
        """
        Initialization of LanguageClassifier class.

        Loads and processes text data from a CSV file. The data is vectorized and
        split into training and test samples. A neural network is then trained
        to categorize the text into languages.

        The CSV file must contain two columns:
        - “Text”: Text data.
        - “Language”: Language labels ('Russian' or 'Italian').
        """
        self.data = pd.read_csv(PATH)

        self.data = self.data[['Text', 'Language']]

        self.data['Language'] = self.data['Language'].map({'Russian': 0, 'Italian': 1})

        self.data = self.data.dropna(subset=['Language'])

        texts = self.data['Text'].values
        labels = self.data['Language'].values

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(texts).toarray()

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        self.model = Sequential([
            Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
            Dense(32, activation='relu'),
            Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

    def predict_language(self, text: str) -> str:
        """
        Determining the language of the entered text.

        Args:
            text (str): The text whose language is to be determined.

        Returns:
            str: The name of the predicted language ('russian' or 'italian').
        """
        text_vectorized = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(text_vectorized)
        predicted_label = np.argmax(prediction)

        languages = {0: 'russian', 1: 'italian'}
        return languages.get(predicted_label, "Unknown")
