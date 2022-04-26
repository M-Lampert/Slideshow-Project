import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os
from glob import glob
from multiprocessing import Pool

import pandas as pd
from framework.activation_layers import ReLU, Sigmoid, Softmax
from framework.input_layer import DefaultInputLayer
from framework.layers import FullyConnectedLayer
from framework.losses import CrossEntropy
from framework.network import NeuralNetwork
from framework.optimizer import SGD
from framework.utils import classes_to_one_hot_vector, get_classification_report, one_hot_vector_to_classes

from src.constants import Paths, class_names, confidence_features, coords_features, features
from src.performance_calculator import calculate_scores
from src.postprocessor import Postprocessor
from src.preprocessor import Preprocessor
from src.utils import load_data, sample_data

# The number of epochs the network should be trained in steps of 10. A value of 10 e.g. trains up to 100 epochs.
number_epochs_in_10s = 10
# The number of runs every choice of parameters should be trained, to counter randomness.
number_runs = 5

# DataFrame with all the choices for all parameters.
trainings_df = pd.DataFrame({"to_delta": ["features", "coords_features"]})
trainings_df = trainings_df.merge(pd.DataFrame({"to_drop": ["features", "coords_features", "confidence_features"]}), how="cross")
trainings_df = trainings_df.merge(pd.DataFrame({"pca": [True, False]}), how="cross")
trainings_df = trainings_df.merge(pd.DataFrame({"augmented": [True, False]}), how="cross")
trainings_df = trainings_df.merge(pd.DataFrame({"run": [run for run in range(number_runs)]}), how="cross")

# The DataFrame the results should be saved to
results_df = pd.DataFrame(columns=list(trainings_df.columns) + ["score", "f1_macro", "precision_macro", "recall_macro", "accuracy"])

string_to_features = {"features": features, "coords_features": coords_features, "confidence_features": confidence_features}


def evaluate_config(i: int) -> pd.DataFrame:
    """
    Helper method that evaluates the i-th config of trainings_df to find the best combination of training data.

    :param i: What config to choose.
    :return: Pandas dataframe with the results.
    """
    all_files = glob(os.path.join(Paths.DATA_MAND_TRAIN_DATA, "*.csv"))
    if trainings_df.loc[i, "augmented"]:
        all_files += glob(os.path.join(Paths.DATA_MAND_TRAIN_DATA / "augmented", "*.csv"))

    preproc = Preprocessor(to_delta=string_to_features[trainings_df.loc[i, "to_delta"]], to_drop=string_to_features[trainings_df.loc[i, "to_drop"]])
    postproc = Postprocessor()

    df_from_each_file = [pd.read_csv(f) for f in all_files]
    x_train, y_labels = preproc.preprocess(df_from_each_file, batch=True, pca=trainings_df.loc[i, "pca"])
    y_train = classes_to_one_hot_vector(y_labels, class_names)

    # sampling:
    x_train, y_train = sample_data(x_train, y_train)

    layers = [
        FullyConnectedLayer(x_train.shape[1], 64),
        ReLU(),
        FullyConnectedLayer(64, 32),
        Sigmoid(),
        FullyConnectedLayer(32, len(class_names)),
        Softmax(),
    ]
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    result_df = trainings_df.iloc[[i]]
    result_df = result_df.merge(pd.DataFrame({"epoch": [(epoch + 1) * 10 for epoch in range(number_epochs_in_10s)]}), how="cross")
    for j in range(result_df.shape[0]):
        # Train the Model
        SGD.update(
            nn=nn,
            loss=CrossEntropy(),
            lr=0.01,
            epochs=10,
            data=(x_train, y_train),
        )

        # Initialize test data
        x_val, y_val = load_data(Paths.DATA_MAND_VAL_DATA, preproc)

        # Test the trained model on so far unseen data
        predictions = nn(x_val, verbose=False)

        classification_report = get_classification_report(predictions=predictions, true_labels=y_val, sparse=False, class_names=class_names)

        events = postproc.postprocess(predictions, batch=True)

        result_df.loc[j, "score"] = calculate_scores(events, one_hot_vector_to_classes(y_val, class_names, sparse=False), verbose=False)
        result_df.loc[j, "f1_macro"] = classification_report.loc[("global", "Macro Avg"), "F1-Score"]
        result_df.loc[j, "precision_macro"] = classification_report.loc[("global", "Macro Avg"), "Precision"]
        result_df.loc[j, "recall_macro"] = classification_report.loc[("global", "Macro Avg"), "Recall"]
        result_df.loc[j, "accuracy"] = classification_report.loc[("global", "Micro Avg / Accuracy"), "F1-Score"]
    return result_df


if __name__ == "__main__":
    n_configs = trainings_df.shape[0]
    with Pool(12) as p:
        for res in p.imap_unordered(evaluate_config, range(n_configs)):
            if res is not None:
                results_df = pd.concat([results_df, res])
    results_df.to_csv("train_evaluation.csv")
