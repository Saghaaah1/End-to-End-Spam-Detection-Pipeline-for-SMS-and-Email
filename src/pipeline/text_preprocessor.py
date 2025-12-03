from typing import Any
import re

from sklearn.feature_extraction.text import CountVectorizer
from utils.config import NB_FEATURES_DEFAULT, DEFAULT_TOKEN_REGEX


class TextPreprocessor:
    """
    Handles raw text cleaning (numbers, case, etc.) and builds the vectorizer.

    This class wraps:
    - a simple preprocessing function (replace numbers, optional lowercase)
    - a CountVectorizer for bag-of-words features
    """

    def __init__(
        self,
        token_pattern: str = DEFAULT_TOKEN_REGEX,
        max_features: int = NB_FEATURES_DEFAULT,
        lowercase: bool = True,
        stop_words: Any = None,
    ):
        self.token_pattern = token_pattern
        self.max_features = max_features
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.vectorizer: CountVectorizer | None = None

    # -------------------------------------------------------------------------
    # Basic message preprocessing
    # -------------------------------------------------------------------------
    def preprocess_message(self, message: str) -> str:
        """
        Apply basic cleaning:

        - Replace any sequence of digits with <NUM>
        - Normalize whitespace
        
        IMPORTANT: no lowercase here -> case handling is done by CountVectorizer.
        """
        # Replace numbers with <NUM>
        message = re.sub(r"\d+", "<NUM>", message)

        # Normalize whitespace: collapse multiple spaces into one
        message = re.sub(r"\s+", " ", message).strip()

        return message

    # -------------------------------------------------------------------------
    # Vectorizer fitting & transformation
    # -------------------------------------------------------------------------
    def _build_vectorizer(self) -> CountVectorizer:
        """
        Create a CountVectorizer configured with our preprocessing and
        tokenization choices.

        'lowercase' here controls case normalization, so we can compare
        lowercase=True vs lowercase=False in experiments.
        """
        # 
        vectorizer = CountVectorizer(
            max_features=self.max_features,
            token_pattern=self.token_pattern,
            stop_words=self.stop_words,
            lowercase=self.lowercase,          
            preprocessor=self.preprocess_message,
        )
        
        return vectorizer

    def fit_vectorizer(self, train_messages):
        """
        Create and fit the CountVectorizer on training messages.

        Returns
        -------
        X_train: sparse matrix
            Feature matrix for training data.
        """
        self.vectorizer = self._build_vectorizer()
        X_train = self.vectorizer.fit_transform(train_messages)
        return X_train

    def transform_messages(self, messages):
        """
        Transform messages into feature matrix using the fitted vectorizer.

        Returns
        -------
        X: sparse matrix
            Feature matrix for the given messages.
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer is not fitted. Call fit_vectorizer() first.")

        X = self.vectorizer.transform(messages)
        return X
