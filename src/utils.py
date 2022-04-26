import os
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from framework.utils import classes_to_one_hot_vector

from src.constants import class_names
from src.preprocessor import Preprocessor


def load_data(
    paths: Union[List[Union[str, Path]], Path, str], preprocessor: Preprocessor, class_names: List[str] = class_names
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a path or multiple paths and preprocess it.

    :param paths: The paths from where to load the data.
    :param preprocessor: The preprocessor instance to use for the preprocessing.
    :param class_names: The classnames of the given data.
    :return: A tuple containing training data and the corresponding ground truth.
    """
    if not isinstance(paths, list):
        paths = [paths]
    all_files = []
    for path in paths:
        all_files += glob(os.path.join(path, "*.csv"))
    df_from_each_file = [pd.read_csv(f) for f in all_files]
    x, y_labels = preprocessor.preprocess(df_from_each_file, batch=True, pca=True)
    y = classes_to_one_hot_vector(y_labels, class_names)
    return x, y


def sample_data(x: np.ndarray, y: np.ndarray, drop_rate: float = 0.2, class_to_sample: str = "idle") -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the same elements of one class in the two given to arrays.

    :param x: The training data
    :param y: The ground truth of the data.
    :param drop_rate: How many of the samples should be dropped. Default: 0.2
    :param class_to_sample: The class that should be sampled. Default: `idle`
    :return: The sampled tuple of training data and ground truth.
    """
    cls_num = class_names.index(class_to_sample)

    rng = np.random.default_rng(seed=42)
    to_drop = []
    for i in range(x.shape[0]):
        if y[i][cls_num] == 1 and rng.uniform() < drop_rate:
            to_drop.append(i)

    x = np.delete(x, to_drop, axis=0)
    y = np.delete(y, to_drop, axis=0)
    return x, y
