"""Hold default values."""
import logging

LOGGING_LEVEL = logging.INFO
DATASET_ATTRS = (
    # Number of Created Discussions
    "NCD_0",
    "NCD_1",
    "NCD_2",
    "NCD_3",
    "NCD_4",
    "NCD_5",
    "NCD_6",
    # Author Increase
    "AI_0",
    "AI_1",
    "AI_2",
    "AI_3",
    "AI_4",
    "AI_5",
    "AI_6",
    # Attention Level
    "AS(NA)_0",
    "AS(NA)_1",
    "AS(NA)_2",
    "AS(NA)_3",
    "AS(NA)_4",
    "AS(NA)_5",
    "AS(NA)_6",
    # Burstiness Level
    "BL_0",
    "BL_1",
    "BL_2",
    "BL_3",
    "BL_4",
    "BL_5",
    "BL_6",
    # Number of Atomic Containers
    "NAC_0",
    "NAC_1",
    "NAC_2",
    "NAC_3",
    "NAC_4",
    "NAC_5",
    "NAC_6",
    # Attention Level (measured with number of contributions)
    "AS(NAC)_0",
    "AS(NAC)_1",
    "AS(NAC)_2",
    "AS(NAC)_3",
    "AS(NAC)_4",
    "AS(NAC)_5",
    "AS(NAC)_6",
    # Contribution Sparseness
    "CS_0",
    "CS_1",
    "CS_2",
    "CS_3",
    "CS_4",
    "CS_5",
    "CS_6",
    # Author Interaction
    "AT_0",
    "AT_1",
    "AT_2",
    "AT_3",
    "AT_4",
    "AT_5",
    "AT_6",
    # Number of Authors
    "NA_0",
    "NA_1",
    "NA_2",
    "NA_3",
    "NA_4",
    "NA_5",
    "NA_6",
    # Average Discussions Length
    "ADL_0",
    "ADL_1",
    "ADL_2",
    "ADL_3",
    "ADL_4",
    "ADL_5",
    "ADL_6",
    # Number of Active Discussion
    "NAD_0",
    "NAD_1",
    "NAD_2",
    "NAD_3",
    "NAD_4",
    "NAD_5",
    "NAD_6",
    # Mean Number of Active Discussion (target)
    "MNAD",
)
DATASET_PREDICT_ATTRS_LEN = len(DATASET_ATTRS) - 1
TARGET_ATTR = "MNAD"
N_FOLD = 5
ASSETS_PATH = "./assets"
DATA_PATH = f"{ASSETS_PATH}/dataset/regression/Twitter/Twitter.data"
RESULTS_PATH = f"{ASSETS_PATH}/results"
CHARTS_PATH = f"{RESULTS_PATH}/charts"
R2 = "R2"
ACCURACY = "Accuracy"
RANK_SIZE = 10
