import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import (
    DATA_PATH,
    SMS_FILE,
    EMAIL_FILE,
    MESSAGE_COL,
    LABEL_COL,
    TRAIN_TEST_SPLIT_SIZE,
    RANDOM_STATE,
)


class DataProcessor:
    """
    Handles loading and splitting SMS and email datasets.
    """

    def __init__(self):
        """Initialize the data processor."""
        self.sms_data = None
        self.email_data = None
    
    
    def load_data(self):
        """
        Load SMS and Email datasets from CSV files.

        Returns:
            Tuple of (sms_df, email_df)
        """
        sms_path = DATA_PATH / SMS_FILE
        email_path = DATA_PATH / EMAIL_FILE

        self.sms_data = pd.read_csv(sms_path, sep=";")
        self.email_data = pd.read_csv(email_path)
        
        # Remove duplicates 
        self.remove_duplicates()
        
        return self.sms_data.copy(), self.email_data.copy()
    

    def remove_duplicates(self):
        """
        Remove exact duplicate rows from SMS and Email datasets
        (same message + same label).
        """
        if self.sms_data is not None:
            self.sms_data = self.sms_data.drop_duplicates(
                subset=[MESSAGE_COL, LABEL_COL],
                ignore_index=True,
            )

        if self.email_data is not None:
            self.email_data = self.email_data.drop_duplicates(
                subset=[MESSAGE_COL, LABEL_COL],
                ignore_index=True,
            )

    def train_eval_sms(self):
        """
        Return train/test sets for SMS-only experiment:
        (train_messages, test_messages, train_labels, test_labels)
        """
        if self.sms_data is None:
            self.load_data()

        sms_messages = self.sms_data[MESSAGE_COL]
        sms_labels = self.sms_data[LABEL_COL]

        train_messages, test_messages, train_labels, test_labels = train_test_split(
            sms_messages,
            sms_labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=sms_labels,
        )

        return train_messages, test_messages, train_labels, test_labels

    def train_eval_email(self):
        """
        Return train/test sets for Email-only experiment:
        (train_messages, test_messages, train_labels, test_labels)
        """
        if self.email_data is None:
            self.load_data()

        email_messages = self.email_data[MESSAGE_COL]
        email_labels = self.email_data[LABEL_COL]

        train_messages, test_messages, train_labels, test_labels = train_test_split(
            email_messages,
            email_labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=email_labels,
        )

        return train_messages, test_messages, train_labels, test_labels

    def train_sms_eval_email(self):
        """
        Cross-domain scenario:
        - Train on ALL SMS messages
        - Test on ALL Email messages

        Returns:
            (train_messages, test_messages, train_labels, test_labels)
        """
        if self.sms_data is None or self.email_data is None:
            self.load_data()

        train_messages = self.sms_data[MESSAGE_COL]
        train_labels = self.sms_data[LABEL_COL]

        test_messages = self.email_data[MESSAGE_COL]
        test_labels = self.email_data[LABEL_COL]

        return train_messages, test_messages, train_labels, test_labels

    def train_eval_combined(self):
        """
        Combine SMS and Email datasets, then create a mixed train/test split.

        Returns:
            (train_messages, test_messages, train_labels, test_labels)
        """
        if self.sms_data is None or self.email_data is None:
            self.load_data()

        combined_df = pd.concat([self.sms_data, self.email_data], axis=0, ignore_index=True)

        messages = combined_df[MESSAGE_COL]
        labels = combined_df[LABEL_COL]

        train_messages, test_messages, train_labels, test_labels = train_test_split(
            messages,
            labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        return train_messages, test_messages, train_labels, test_labels


    def balance_training_data(self, training_messages, training_labels):
        """
        Oversample the minority class so that both classes have the same size.

        Returns
        -------
        (balanced_messages, balanced_labels)
        """
        training_data = pd.concat(
            [training_messages.reset_index(drop=True),
             training_labels.reset_index(drop=True)],
            axis=1
        )

        counts = training_labels.value_counts()
        if counts[0] == counts[1]:
            return training_messages, training_labels  # already balanced

        if counts[1] > counts[0]:
            minority_label = 0
            diff = counts[1] - counts[0]
        else:
            minority_label = 1
            diff = counts[0] - counts[1]

        minority_rows = training_data[training_data[LABEL_COL] == minority_label]

        # sample with replacement to match the majority count
        extra_samples = minority_rows.sample(n=diff, replace=True, random_state=RANDOM_STATE)
        balanced_data = pd.concat([training_data, extra_samples], axis=0).reset_index(drop=True)

        balanced_messages = balanced_data[MESSAGE_COL]
        balanced_labels = balanced_data[LABEL_COL]

        return balanced_messages, balanced_labels

    
    def preprocess_data(self, scenario="sms_only", balance=False):
        """
        Preprocessing for text data.

        - ensures data is loaded
        - creates train/test split according to the scenario
        - optionally balances the training set

        Returns
        -------
        (train_messages, test_messages, train_labels, test_labels)
        """
        if self.sms_data is None or self.email_data is None:
            self.load_data()

        if scenario == "sms_only":
            train_messages, test_messages, train_labels, test_labels = self.train_eval_sms()
        elif scenario == "email_only":
            train_messages, test_messages, train_labels, test_labels = self.train_eval_email()
        elif scenario == "sms_to_email":
            train_messages, test_messages, train_labels, test_labels = self.train_sms_eval_email()
        elif scenario == "combined":
            train_messages, test_messages, train_labels, test_labels = self.train_eval_combined()
        else:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                "Expected one of: sms_only, email_only, sms_to_email, combined."
                )

        if balance:
            train_messages, train_labels = self.balance_training_data(
                train_messages, train_labels
            )

        return train_messages, test_messages, train_labels, test_labels

    def load_and_preprocess(self, scenario="sms_only", **preprocess_kwargs):
        """
        Convenience method to load and preprocess in one step.
        """
        
        self.load_data()
        return self.preprocess_data(scenario=scenario, **preprocess_kwargs)





