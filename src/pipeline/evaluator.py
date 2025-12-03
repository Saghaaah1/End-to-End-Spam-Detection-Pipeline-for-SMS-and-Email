from typing import Dict


class Evaluator:
    """
    Computes evaluation metrics: accuracy, precision, recall, f1,
    and extracts confusion matrix elements.
    """

    def __init__(self, pos_label=1):
        self.pos_label = pos_label

    # -------------------------
    # Basic metrics
    # -------------------------
    def accuracy(self, truth, pred) -> float:
        """Accuracy = correct predictions / total."""
        truth = list(truth)
        pred = list(pred)

        correct = 0
        for t, p in zip(truth, pred):
            if t == p:
                correct += 1

        return correct / len(truth) if truth else 0.0

    def precision(self, truth, pred) -> float:
        """
        Precision = TP / (TP + FP)
        Among predicted positives, how many are truly positive?
        """
        truth = list(truth)
        pred = list(pred)

        tp = 0
        fp = 0

        for t, p in zip(truth, pred):
            if p == self.pos_label:           # model predicts positive
                if t == self.pos_label:       # correct
                    tp += 1
                else:                         # predicted positive but actually negative
                    fp += 1

        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, truth, pred) -> float:
        """
        Recall = TP / (TP + FN)
        Among actual positives, how many did we catch?
        """
        truth = list(truth)
        pred = list(pred)

        tp = 0
        fn = 0

        for t, p in zip(truth, pred):
            if t == self.pos_label:           # actually positive
                if p == self.pos_label:       # correctly predicted
                    tp += 1
                else:                         # missed positive
                    fn += 1

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1_score(self, truth, pred) -> float:
        """F1 = harmonic mean of precision and recall."""
        prec = self.precision(truth, pred)
        rec = self.recall(truth, pred)

        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    # -------------------------
    # Confusion matrix elements
    # -------------------------
    def confusion_elements(self, truth, pred) -> Dict[str, int]:
        """
        Return dict with keys: 'tp', 'tn', 'fp', 'fn'.
        """
        truth = list(truth)
        pred = list(pred)

        tp = tn = fp = fn = 0

        for t, p in zip(truth, pred):
            if t == self.pos_label and p == self.pos_label:
                tp += 1
            elif t != self.pos_label and p != self.pos_label:
                tn += 1
            elif t != self.pos_label and p == self.pos_label:
                fp += 1
            elif t == self.pos_label and p != self.pos_label:
                fn += 1

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
