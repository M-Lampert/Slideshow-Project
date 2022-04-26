import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from framework.hyperparameter_search import HyperparameterSearch
from framework.utils import one_hot_vector_to_classes

from src.constants import Paths, class_names, coords_features, features
from src.performance_calculator import calculate_scores
from src.postprocessor import Postprocessor
from src.preprocessor import Preprocessor
from src.utils import load_data


def custom_score_func(predictions, true_labels, sparse):
    """
    Wrapper so that the score function can be used inside the frameworks hyperparameter tuning.
    """
    postproc = Postprocessor(12, 0.8, class_names, thresholds={"swipe_left": 0.4, "swipe_right": 0.4, "rotate": 0.6})
    events = postproc.postprocess(predictions, batch=True)
    return calculate_scores(events, one_hot_vector_to_classes(true_labels, class_names, sparse=False), verbose=False)


if __name__ == "__main__":
    preproc = Preprocessor(Paths.MODELS / "pca.pkl", to_delta=features, to_drop=coords_features)

    x_train, y_train = load_data([Paths.DATA_MAND_TRAIN_DATA, Paths.DATA_MAND_TRAIN_DATA / "augmented"], preproc)
    x_val, y_val = load_data(Paths.DATA_MAND_VAL_DATA, preproc)
    gt = pd.DataFrame(one_hot_vector_to_classes(y_val, class_names, sparse=False), columns=["ground_truth"])

    search = HyperparameterSearch((x_train, y_train), (x_val, y_val), [("Score", custom_score_func), "Accuracy", "F1", "Precision", "Recall"])

    search(
        cores=12,
        n_layers=[3, 5],
        learning_rates=[0.1, 0.01],
        losses=["crossentropy"],
        activations=["relu", "sigmoid"],
        n_neurons=[32, 64],
        n_epochs=[20],
        path_to_csv="parameter_search.csv",
    )
