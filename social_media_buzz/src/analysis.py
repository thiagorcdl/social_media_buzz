"""Module for implementing the overall logics.

This consists of extracting data, training model, using it and ranking the
results.
"""
import logging

from tqdm import tqdm

from social_media_buzz.src.constants import RANK_SIZE
from social_media_buzz.src.data import (
    get_candidate_features, prepare_dataset,
    show_results,
)
from social_media_buzz.src.linear_regression import LinearRegressionModel

logger = logging.getLogger(__name__)


def rank_features(results, desc, top=RANK_SIZE) -> list:
    """Get the top most significative features by averaging out their results.
    """
    analysis = {}
    amount = len(results)

    for fold_results in tqdm(results, desc=desc):
        for result in fold_results:
            attr_name = result[0]
            analysis[attr_name] = analysis.setdefault(attr_name, 0) + result[1]

    averages = map(lambda x: (x[0], x[1] / amount), list(analysis.items()))
    ranking = sorted(list(averages), key=lambda x: x[1] * -1)
    return ranking[:top]


def main():
    """Run main logics for comparing features."""
    features = get_candidate_features()
    r_results = []
    acc_results = []

    for training_data, testing_data in prepare_dataset():
        fold_r_results = []
        fold_acc_results = []
        model = LinearRegressionModel(training_data)
        progress = tqdm(features, position=1)
        for feature in progress:
            progress.set_description(f"Trying feature {feature}")
            model.train(feature)
            model.test(testing_data)
            fold_r_results.append((feature, model.r_squared))
            fold_acc_results.append((feature, model.testing_acc))

        r_results.append(fold_r_results)
        acc_results.append(fold_acc_results)

    for name, results in (("R-Squared", r_results), ("Accuracy", acc_results)):
        rank = rank_features(results, f"Processing {name} results.")
        logger.info(f"{name} ranking:")
        show_results(rank, results)
