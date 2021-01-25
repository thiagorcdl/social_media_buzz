"""Module for implementing the logics behind the Linear Regression model
to be applied on the Testing/Validation data.
"""
from scipy import stats

from social_media_buzz.src.constants import DEFAULT_TARGET_ATTR
from social_media_buzz.src.data import get_column


class LinearRegressionModel:
    """Class used to represent a linear regression model."""

    def __init__(self, training_data):
        self.trained: bool = False
        self.training_data = training_data
        self.testing_data = []
        self.testing_result = []
        self.slope = 0
        self.intercept = 0
        self.r = 0
        self.p = 0
        self.std_err = 0

    @property
    def r_squared(self):
        """Return the correlation value (R-squared)."""
        return self.r ** 2

    def train(self, predictor_attr, target_attr=DEFAULT_TARGET_ATTR):
        """Use self.training_data to train a model using given feature."""
        x_axis = get_column(self.training_data, predictor_attr)
        y_axis = get_column(self.training_data, target_attr)

        # Run actual linear regression
        slope, intercept, r, p, std_err = stats.linregress(x_axis, y_axis)

        self.slope = slope
        self.intercept = intercept
        self.r = r
        self.p = p
        self.std_err = std_err
        self.trained = True

    def predict(self, x):
        """Predict y value for given x."""
        return self.slope * x + self.intercept

    @property
    def testing_err(self):
        """Return difference between predicted and actual target value."""
        diffs = []

        for pred, actual in zip(self.result, self.testing_result):
            diffs.append(abs(actual - pred))

        return sum(diffs)

    def test(self, testing_data):
        """Use training_data to evaluate correlation.

        Return R-squared.
        """
        if not self.trained:
            raise ValueError("Model hasn't been trained yet.")

        self.testing_data = testing_data
        self.testing_result = list(map(self.predict, testing_data))
        return self.result
