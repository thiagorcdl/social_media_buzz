"""Module for implementing the overall logics.

This consists of extracting data, training model, using it and ranking the
results.
"""
from social_media_buzz.src.data import (
    get_candidate_features, prepare_dataset,
    show_results,
)
from social_media_buzz.src.linear_regression import LinearRegressionModel


def rank_features(results, top=10) -> list:
    """Get the top most significative features.

    Features are ranked by squaring the indexes. Higher is better.
    """
    analysis = {}

    for fold_results in results:
        # Higher results are placed last in the array so they have more
        # significative indexes
        fold_results = sorted(fold_results, key=lambda x: x[0])

        for idx, result in enumerate(fold_results):
            analysis[result[1]] = analysis.setdefault(result[1], 0) + idx ** 2

    return sorted(list(analysis.items()), key=lambda x: x[1] * -1)[:top]


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
            model.train(feature)
            model.test(testing_data)
            fold_r_results.append((model.r_squared, feature))
            fold_acc_results.append((model.testing_err, feature))

        r_results.append(fold_r_results)
        acc_results.append(fold_acc_results)

    r_rank = rank_features(r_results)
    acc_rank = rank_features(acc_results)
    show_results(r_rank)
    show_results(acc_rank)
