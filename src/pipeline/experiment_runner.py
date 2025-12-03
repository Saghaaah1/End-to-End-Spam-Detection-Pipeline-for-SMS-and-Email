from typing import Type, Any, Dict

from pipeline.data_processor import DataProcessor
from pipeline.text_preprocessor import TextPreprocessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import (
    DEFAULT_TOKEN_REGEX,
    NB_FEATURES_DEFAULT,
    NB_ITERATIONS_DEFAULT,
)
from utils.utils import print_step_header, print_results_summary

class ExperimentRunner:
    """
    Orchestrates full experiments:
    - chooses dataset scenario (sms_only, email_only, combined, sms_to_email)
    - runs preprocessing, training, evaluation
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        text_preprocessor_cls: Type[TextPreprocessor] = TextPreprocessor,
        model_trainer_cls: Type[ModelTrainer] = ModelTrainer,
        evaluator_cls: Type[Evaluator] = Evaluator,
    ):
        self.data_processor = data_processor
        self.text_preprocessor_cls = text_preprocessor_cls
        self.model_trainer_cls = model_trainer_cls
        self.evaluator_cls = evaluator_cls

    def _get_data_for_scenario(self, scenario: str):
        """
        Map scenario string to appropriate DataProcessor method.
        Returns (train_messages, test_messages, train_labels, test_labels)
        """
        scenario = scenario.lower()

        if scenario == "sms_only":
            return self.data_processor.train_eval_sms()
        elif scenario == "email_only":
            return self.data_processor.train_eval_email()
        elif scenario == "sms_to_email":
            return self.data_processor.train_sms_eval_email()
        elif scenario == "combined":
            return self.data_processor.train_eval_combined()
        else:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Expected one of: sms_only, email_only, sms_to_email, combined."
            )

    def run_experiment(
        self,
        scenario: str,
        token_pattern: str = DEFAULT_TOKEN_REGEX,
        max_features: int = NB_FEATURES_DEFAULT,
        lowercase: bool = True,
        stop_words: Any = None,
        max_iter: int = NB_ITERATIONS_DEFAULT,
    ) -> Dict[str, Any]:
        """
        Run a full experiment for a given scenario and return metrics.

        Parameters
        ----------
        scenario : str
            One of 'sms_only', 'email_only', 'sms_to_email', 'combined'
        token_pattern : str
            Regex for tokenization in CountVectorizer
        max_features : int
            Vocabulary size
        lowercase : bool
            Whether to lowercase text during preprocessing
        stop_words : any
            Stopword setting for CountVectorizer (None or 'english')
        max_iter : int
            Max iterations for Logistic Regression

        Returns
        -------
        dict
            Dictionary with metrics, confusion matrix elements and config used.
        """
        print_step_header(1, f"Running experiment: {scenario}")

        # 1) Get data
        train_m, test_m, train_y, test_y = self._get_data_for_scenario(scenario)

        # 2) Text preprocessing
        print_step_header(2, "Building text features")
        preprocessor = self.text_preprocessor_cls(
            token_pattern=token_pattern,
            max_features=max_features,
            lowercase=lowercase,
            stop_words=stop_words,
        )
        X_train = preprocessor.fit_vectorizer(train_m)
        X_test = preprocessor.transform_messages(test_m)

        # 3) Model training
        print_step_header(3, "Training model")
        trainer = self.model_trainer_cls(max_iter=max_iter)
        trainer.build_model()
        trainer.train(X_train, train_y)

        # 4) Evaluation
        print_step_header(4, "Evaluating model")
        evaluator = self.evaluator_cls(pos_label=1)
        pred = trainer.predict(X_test)

        acc = evaluator.accuracy(test_y, pred)
        prec = evaluator.precision(test_y, pred)
        rec = evaluator.recall(test_y, pred)
        f1 = evaluator.f1_score(test_y, pred)
        cm = evaluator.confusion_elements(test_y, pred)

        results = {
            "scenario": scenario,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": cm["tp"],
            "tn": cm["tn"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "n_train": len(train_m),
            "n_test": len(test_m),
            "token_pattern": token_pattern,
            "max_features": max_features,
            "lowercase": lowercase,
            "stop_words": stop_words,
            "max_iter": max_iter,
        }

        print_results_summary(results)
        return results
