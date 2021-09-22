import numpy as np


class BaseClassifierScorer(object):
    _name = "base"

    def __call__(self, predictions, labels):
        raise NotImplementedError()

    @property
    def name(cls):
        return cls._name


class Accuracy(BaseClassifierScorer):
    _name = "Accuracy"

    def __call__(self, predictions, labels):
        return (predictions == labels).mean()


class F1(BaseClassifierScorer):
    _name = "F-1 score"

    def __call__(self, predictions, labels):
        # True positives
        tp = np.logical_and(predictions == 1, labels == 1).sum()
        # Precision
        P = tp / (predictions == 1).sum()
        # Recall
        R = tp / (labels == 1).sum()
        # F-score
        return 2 * P * R / (P + R)


class Matthews(BaseClassifierScorer):
    _name = "Matthew's correlation"

    def __call__(self, predictions, labels):
        # True/False positives/negatives
        tp = np.logical_and(predictions == 1, labels == 1).sum()
        fp = np.logical_and(predictions == 1, labels == 0).sum()
        tn = np.logical_and(predictions == 0, labels == 0).sum()
        fn = np.logical_and(predictions == 0, labels == 1).sum()
        # Correlation coefficient
        m = (tp * tn) - (fp * fn)
        m /= np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-20

        return m
