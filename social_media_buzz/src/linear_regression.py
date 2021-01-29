"""Module for implementing the logics behind the Linear Regression model
to be applied on the Testing/Validation data.
"""
import matplotlib.pyplot as plt
from scipy import stats

from social_media_buzz.src.constants import (
    CHARTS_PATH, COLOR_1, COLOR_2,
    TARGET_ATTR,
)
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
        self.predictor_attr = ""
        self.target_attr = ""

    @property
    def r_squared(self):
        """Return the correlation value (R-squared)."""
        return self.r ** 2

    def train(self, predictor_attr, target_attr=TARGET_ATTR):
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
        self.predictor_attr = predictor_attr
        self.target_attr = target_attr
        self.trained = True

    def predict(self, x):
        """Predict y value for given x."""
        return self.slope * x + self.intercept

    @property
    def testing_err(self):
        """Return average difference between predicted and actual target value.
        """
        diffs = []

        target_axis = get_column(self.testing_data, self.target_attr)
        for pred, actual in zip(self.testing_result, target_axis):
            diffs.append(abs(actual - pred))

        return sum(diffs) / len(self.testing_result)

    @property
    def testing_acc(self):
        """Return the inverse of the error."""
        return (1 / self.testing_err) if self.testing_err else 1

    def test(self, testing_data):
        """Use testing_data to determine whether predictions of the trained
        model (using previously specified predictor_attr) are accurate.
        """
        if not self.trained:
            raise ValueError("Model hasn't been trained yet.")

        self.testing_data = testing_data
        predictor_axis = get_column(testing_data, self.predictor_attr)
        self.testing_result = list(map(self.predict, predictor_axis))
        return self.testing_result

    def plot_chart(self, filename="chart", show=False, save=True):
        """Plot lines using test data and function from model."""
        fig, ax = plt.subplots()
        ax.set(xlabel=self.predictor_attr, ylabel=self.target_attr)
        x_axis = get_column(self.testing_data, self.predictor_attr)
        y_axis = get_column(self.testing_data, self.target_attr)
        plt.scatter(x_axis, y_axis, color=COLOR_1)
        plt.plot(x_axis, self.testing_result, color=COLOR_2)
        if show:
            plt.show()
        if save:
            plt.savefig(f"{CHARTS_PATH}/{filename}.png")
        plt.clf()
