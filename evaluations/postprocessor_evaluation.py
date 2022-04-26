import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Union

import pandas as pd
from framework.network import NeuralNetwork
from framework.utils import one_hot_vector_to_classes
from tqdm import tqdm

from src.constants import Paths, class_names, class_names_opt, coords_features, features
from src.performance_calculator import calculate_scores
from src.postprocessor import Postprocessor
from src.preprocessor import Preprocessor
from src.utils import load_data


def evaluate(
    out_path: Union[str, Path],
    class_names: List[str] = class_names,
    window_sizes=None,
    gesture_thresholds=None,
    idle_thresholds=None,
    nn: NeuralNetwork = NeuralNetwork.load(Paths.MODELS / "network.pkl"),
    preproc: Preprocessor = Preprocessor(Paths.MODELS / "pca.pkl", to_delta=features, to_drop=coords_features),
    val_data_paths: Union[str, Path, List[str], List[Path]] = Paths.DATA_MAND_VAL_DATA,
):
    """
    Method to find the exact values for the sliding window approach of our postprocessing.

    :param out_path: Destination path of the evaluation results csv file
    :param class_names: List of possible class names, e.g. mandatory: ["idle", "rotate", "swipe_left", "swipe_right"]
    :param window_sizes: Possible window sizes to be analysed
    :param gesture_thresholds: Possible gesture thresholds to be analysed
    :param idle_thresholds: Possible idle thresholds to be analysed
    :param nn: A trained neural network
    :param preproc: A preprocessor to preprocess the validation data
    :param val_data_paths: One or more data_paths in a list
    :return: None (saves results in a csv file to out_path)
    """
    if idle_thresholds is None:
        idle_thresholds = [x / 10 for x in range(7, 11)]
    if gesture_thresholds is None:
        gesture_thresholds = [x / 10 for x in range(2, 9)]
    if window_sizes is None:
        window_sizes = list(range(6, 30, 2))

    # DataFrame with all the choices for all parameters.
    configurations_df = pd.DataFrame({"class_name": [cn for cn in class_names if cn != "idle"]})
    configurations_df = configurations_df.merge(pd.DataFrame({"window_size": window_sizes}), how="cross")
    configurations_df = configurations_df.merge(pd.DataFrame({"thresh": gesture_thresholds}), how="cross")
    configurations_df = configurations_df.merge(pd.DataFrame({"idle_thresh": idle_thresholds}), how="cross")
    configurations_df["score"] = None

    # Load validation data
    x_val, y_val = load_data(val_data_paths, preproc, class_names=class_names)
    gt = one_hot_vector_to_classes(y_val, class_names, sparse=False)

    # Calculate predictions once
    predictions = nn(x_val)

    # Calculate individual scores for the different gestures and write them into the dataframe
    for i in tqdm(range(configurations_df.shape[0])):
        postproc = Postprocessor(
            window_size=configurations_df.iloc[i]["window_size"].item(),
            idle_thresh=configurations_df.iloc[i]["idle_thresh"],
            thresholds={k: (configurations_df.iloc[i]["thresh"] if k == configurations_df.iloc[i]["class_name"] else 1) for k in class_names if k != "idle"},
            classes=class_names,
        )
        events = postproc.postprocess(predictions, batch=True)
        configurations_df.loc[i, "score"] = calculate_scores(events, gt, verbose=False, gestures=configurations_df.iloc[i]["class_name"])
    # To avoid dtype errors with score column
    configurations_df = configurations_df.astype({"score": "float64"})
    # Only keep the best threshold settings for the different gestures
    score_maximum_row_indices = configurations_df.groupby(["window_size", "idle_thresh", "class_name"])["score"].idxmax()
    configurations_df = configurations_df.iloc[score_maximum_row_indices]

    # Merge results for the different gestures
    results_dfs = []
    for class_name in [cn for cn in class_names if cn != "idle"]:
        results_dfs.append(
            configurations_df[configurations_df["class_name"] == class_name][["window_size", "idle_thresh", "thresh"]].rename(
                columns={"thresh": f"{class_name}_thresh"}
            )
        )
    configurations_df = results_dfs[0]
    for i in range(1, len(results_dfs)):
        configurations_df = configurations_df.merge(results_dfs[i], left_on=["window_size", "idle_thresh"], right_on=["window_size", "idle_thresh"])

    # The final score
    configurations_df["score"] = None

    # Calculate final scores for all gestures together
    for i in tqdm(range(configurations_df.shape[0])):
        postproc = Postprocessor(
            window_size=configurations_df.iloc[i]["window_size"].item(),
            idle_thresh=configurations_df.iloc[i]["idle_thresh"],
            thresholds={k: configurations_df.iloc[i][k + "_thresh"] for k in [cn for cn in class_names if cn != "idle"]},
            classes=class_names,
        )
        events = postproc.postprocess(predictions, batch=True)
        configurations_df.loc[i, "score"] = calculate_scores(events, gt, verbose=False, gestures="all")

    configurations_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    evaluate(
        "postprocessor_evaluation_opt.csv",
        class_names=class_names_opt,
        nn=NeuralNetwork.load(Paths.MODELS / "network_opt.pkl"),
        preproc=Preprocessor(to_delta=features, to_drop=coords_features, pca_path=Paths.MODELS / "pca_opt.pkl"),
        val_data_paths=[Paths.DATA_MAND_VAL_DATA, Paths.DATA_OPT_VAL_DATA],
    )
