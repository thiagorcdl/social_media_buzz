"""Module for implementing the overall logics.

This consists of extracting data, training model, using it and ranking the
results.
"""
import logging

from social_media_buzz.src.data import (
    get_candidate_features, prepare_dataset,
    show_results,
)
from social_media_buzz.src.linear_regression import LinearRegressionModel

logger = logging.getLogger(__name__)


def rank_features(results, top=10) -> list:
    """Get the top most significative features.

    Features are ranked by squaring the indexes. Higher is better.
    """
    analysis = {}

    for fold_results in results:
        # Higher results are placed last in the array so they have more
        # significative indexes
        fold_results = sorted(fold_results, key=lambda x: x[1])
        logger.debug(fold_results)

        for idx, result in enumerate(fold_results):
            analysis[result[0]] = analysis.setdefault(result[0], 0) + idx ** 2

    ranking = sorted(list(analysis.items()), key=lambda x: x[1] * -1)[:top]
    return list(map(lambda x: x[0], ranking))


def main():
    """Run main logics for comparing features."""
    features = get_candidate_features()
    r_results = []
    acc_results = []

    for training_data, testing_data in prepare_dataset():
        fold_r_results = []
        fold_acc_results = []
        model = LinearRegressionModel(training_data)

        for feature in features:
            logger.debug(f"Trying feature {feature}.")
            model.train(feature)
            model.test(testing_data)
            fold_r_results.append((feature, model.r_squared))
            fold_acc_results.append((feature, model.testing_err))

        r_results.append(fold_r_results)
        acc_results.append(fold_acc_results)

    for name, results in (("R-Squared", r_results), ("Accuracy", acc_results)):
        logger.info(f"Processing {name} results.")
        rank = rank_features(results)
        logger.info(f"{name} ranking:")
        show_results(rank, results)
