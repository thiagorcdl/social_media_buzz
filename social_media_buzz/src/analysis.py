"""Module for implementing the overall logics.

This consists of extracting data, training model, using it and ranking the
results.
"""
import logging
from collections import defaultdict

from tqdm import tqdm

from social_media_buzz.src.constants import ACCURACY, R2, RANK_SIZE
from social_media_buzz.src.data import (
    get_candidate_features, prepare_dataset,
    show_rank, write_results,
)
from social_media_buzz.src.linear_regression import LinearRegressionModel

logger = logging.getLogger(__name__)


def rank_features(metric_result, name, top=RANK_SIZE) -> list:
    """Get the top most significative features by averaging out their results.
    """
    analysis = defaultdict(lambda: 0)
    amount = len(metric_result)

    for fold_result in tqdm(metric_result, desc=f"Processing {name} results."):
        for attr_result in fold_result:
            attr_name = attr_result[0]
            analysis[attr_name] += attr_result[1]

    averages = map(lambda x: (x[0], x[1] / amount), list(analysis.items()))
    ranking = sorted(list(averages), key=lambda x: x[1] * -1)
    return ranking[:top]


def get_ranks(fold_results=None):
    """Print ranks to terminal and write csv files."""
    results = fold_results

    for name in (R2, ACCURACY):
        metric_result = results.get(name)
        rank = rank_features(metric_result.values(), name)
        logger.info(f"{name} ranking:")
        show_rank(rank, metric_result, name)


def main():
    """Run main logics for comparing features.

    For each fold, for each attribute, train model using that attribute and
    the target feature. Then, calculate R-squared, accuracy and store them.
    Write results in CSV files and rank the best attributes for each metric.
    """
    features = get_candidate_features()
    results = defaultdict(lambda: defaultdict(list))
    fold_results = defaultdict(lambda: defaultdict(list))

    for idx, dataset in enumerate(prepare_dataset()):
        training_data, testing_data = dataset
        model = LinearRegressionModel(training_data)
        progress = tqdm(features, position=1)
        for attr_name in progress:
            progress.set_description(f"Trying feature {attr_name}")
            model.train(attr_name)
            model.test(testing_data)
            results[R2][attr_name].append(model.r_squared)
            results[ACCURACY][attr_name].append(model.testing_acc)
            fold_results[R2][idx].append((attr_name, model.r_squared))
            fold_results[ACCURACY][idx].append((attr_name, model.testing_acc))

    write_results(results)
    get_ranks(fold_results)
