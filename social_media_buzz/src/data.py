"""Module for implementing the data preparation.

This involves dividing the known data set into Training and Testing subsets.
"""
import logging
from functools import lru_cache

from social_media_buzz.src.constants import (
    DEFAULT_DATA_PATH, DEFAULT_N_FOLD,
    DATASET_ATTRS,
)

logger = logging.getLogger(__name__)


def load_dataset(file_path=DEFAULT_DATA_PATH):
    """Open dataset file and load ITS contents."""
    dataset = []

    logger.info(f"Loading dataset from {file_path}.")
    with open(file_path, "r") as file:
        for line in file.readlines():
            new_line = list(map(float, line.split(",")))
            dataset.append(new_line)

    return dataset


def get_candidate_features() -> list:
    """Return list of input features to be used as predictor candidates."""
    return DATASET_ATTRS[:-1]


def prepare_dataset(n_fold=DEFAULT_N_FOLD) -> tuple:
    """Yield list of Training and Testing data tuples, respectively.

    Use cross-validation / folding technique to train and test models
    based on different combination of data, to prevent overfitting.
    """
    dataset = load_dataset()
    chunk_size = len(dataset) // n_fold
    logger.info(f"Dividing dataset in {n_fold} folds.")

    for fold in range(n_fold):
        testing_init = fold * chunk_size
        testing_end = testing_init + chunk_size
        logger.debug(f"Yielding folds {fold} for testing.")

        training_data = dataset[:testing_init]
        testing_data = dataset[testing_init:testing_end]
        training_data += dataset[testing_end:]
        yield training_data, testing_data


@lru_cache(maxsize=128)
def get_attr_idx(attr) -> int:
    """Return index of the column for given attribute in the data set."""
    return DATASET_ATTRS.index(attr)


def get_column(dataset, attr) -> list:
    """Return list of values for column `attr`."""
    idx = get_attr_idx(attr)
    return [row[idx] for row in dataset]


def show_results(rank, results):
    """Print rank."""
    for feature in rank:
        for result in results:
            if feature != result[0]:
                continue
            print(f"{feature}: {result[1]}")
            break
