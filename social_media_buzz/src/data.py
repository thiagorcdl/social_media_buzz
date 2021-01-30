"""Module for implementing the data preparation.

This involves dividing the known data set into Training and Testing subsets.
"""
import csv
import logging
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from social_media_buzz.src.constants import (
    CHARTS_PATH, DATASET_PREDICT_ATTRS_LEN, DATA_PATH, N_FOLD,
    DATASET_ATTRS, RESULTS_PATH,
)

logger = logging.getLogger(__name__)


def create_dirs():
    """Ensure necessary paths are valid."""
    for dir_path in [RESULTS_PATH, CHARTS_PATH]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_dataset(file_path=DATA_PATH):
    """Open dataset file and load its contents."""
    dataset = []

    desc = f"Loading dataset from {file_path}"
    with open(file_path, "r") as file:
        for line in tqdm(file.readlines(), desc=desc):
            new_line = list(map(float, line.split(",")))
            dataset.append(new_line)

    return dataset


def prepare_dataset(n_fold=N_FOLD) -> tuple:
    """Yield list of Training and Testing data tuples, respectively.

    Use cross-validation / folding technique to train and test models
    based on different combination of data, to prevent overfitting.
    """
    create_dirs()
    dataset = load_dataset()
    chunk_size = len(dataset) // n_fold
    progress = tqdm(range(n_fold), position=0)

    for fold in progress:
        progress.set_description(f"Yielding fold {fold} for testing")
        testing_init = fold * chunk_size
        testing_end = testing_init + chunk_size

        training_data = dataset[:testing_init]
        testing_data = dataset[testing_init:testing_end]
        training_data += dataset[testing_end:]
        yield training_data, testing_data


@lru_cache(maxsize=1)
def get_candidate_features() -> list:
    """Return list of input features to be used as predictor candidates."""
    return DATASET_ATTRS[:DATASET_PREDICT_ATTRS_LEN]


@lru_cache(maxsize=128)
def get_attr_idx(attr) -> int:
    """Return index of the column for given attribute in the data set."""
    return DATASET_ATTRS.index(attr)


def get_column(dataset, attr) -> list:
    """Return list of values for column `attr`."""
    idx = get_attr_idx(attr)
    return [row[idx] for row in dataset]


def show_rank(rank, results, name):
    """Print specific metric's rank to terminal and write CSV file."""
    logger.debug(rank)
    logger.debug(results)
    print_lines = [""]
    path = f"{RESULTS_PATH}/{name.lower()}_rank.csv"

    with open(path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Rank", "Attribute", f"Average {name}"])

        for idx, item in enumerate(rank, start=1):
            print_lines.append(f"{idx:02} - {item[0]}: {item[1]}")
            csv_writer.writerow([idx, item[0], item[1]])

        logger.info("\n".join(print_lines))


def write_results(results):
    """Write results in CSV file.

    Build headers from amount of folds present in attribute results.
    """

    for name, metric_content in results.items():
        csv_headers = ["attr"] + [
            f"fold{idx:02}" for idx
            in range(len(list(metric_content.values())[0]))
        ]
        path = f"{RESULTS_PATH}/{name.lower()}_results.csv"

        with open(path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_headers)
            for attr, folds in metric_content.items():
                csv_writer.writerow([attr] + folds)
