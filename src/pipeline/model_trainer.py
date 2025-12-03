from typing import Any

from sklearn.linear_model import LogisticRegression

from utils.config import NB_ITERATIONS_DEFAULT


class ModelTrainer:
    """
    Trains and stores the classification model (e.g. Logistic Regression).
    """

    def __init__(self, max_iter: int = NB_ITERATIONS_DEFAULT):
        self.max_iter = max_iter
        self.model: LogisticRegression | None = None

    def build_model(self):
        """Instantiate the Logistic Regression model."""
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            # Add other hyperparams later (C, penalty, solver...)
        )
        return self.model

    def train(self, X_train, y_train):
        """Fit the model on training data."""
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Generate predictions from the trained model."""
        if self.model is None:
            raise RuntimeError("Model is not built/trained yet.")
        return self.model.predict(X_test)
