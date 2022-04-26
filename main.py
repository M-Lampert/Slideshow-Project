import argparse
import multiprocessing
from pathlib import Path

import pandas as pd
import requests
from framework.network import NeuralNetwork
from slideshow import slideshow_server

from src import data_processor as dp
from src.constants import Paths, class_names, class_names_opt, coords_features, event_dict, features
from src.postprocessor import Postprocessor
from src.preprocessor import Preprocessor


def process(q: multiprocessing.Queue, neural_net: NeuralNetwork, preprocessor: Preprocessor, postprocessor: Postprocessor, src: Path = None, dest: Path = None):
    """
    The process of the controller that predicts the events and controls the server or writes to csv.

    :param q: The Queue to use for the multiprocessing.
    :param neural_net: The neural network to use for the predictions.
    :param preprocessor: The preprocessor to use.
    :param postprocessor: The postprocessor to use.
    :param src: The source-path of a .csv-file of a video transcript that should be tested. If a path is given, the server won't be controlled, since this is only necessary in live mode. Default: None
    :param dest: The destination-path to save the .csv-file with events to. Default: None
    :return: None
    """
    # Starts with one event, because the preprocessor drops the first frame,
    # because it has to compute the difference of two frames, which is impossible at the start.
    events = ["idle"]
    while True:
        curr_frame = q.get()
        if curr_frame is not None:
            x_preproc = preprocessor.preprocess(curr_frame, pca=True)
            if x_preproc is not None:
                pred = neural_net(x_preproc, batch=False, verbose=False)
                event = postprocessor.postprocess(pred)
                # If no source path is given, we are in live mode and want to control the server
                if src is None:
                    if event != "idle":
                        print(f"{event}")
                        requests.get(f"http://localhost:8000/send_event?event={event_dict[event]}")
                # If a path is given we are in test mode and want to save the events
                else:
                    events.append(event)
        else:
            if src:
                csv = pd.read_csv(src)
                csv["events"] = events
                csv.to_csv(dest, index=False)
            q.close()
            break


def get_args():
    """
    Argument parser. The two commands are:
        - `python main.py --live --GESTURES_TO_USE`
        - `python main.py --test --GESTURES_TO_USE --src SRC_PATH --dest DEST_PATH`

    :return: The parsed arguments
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action=argparse.BooleanOptionalAction, help="Starts the live mode including the presentation server.")
    group.add_argument("--test", action=argparse.BooleanOptionalAction, help="Starts the test mode. Requires a source and destination path to be specified.")

    group = parser.add_argument_group("Test mode paths")
    group.add_argument("--src", type=Path, help="Path to source csv.")
    group.add_argument("--dest", type=Path, help="Path to save the predictions to.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mand", action=argparse.BooleanOptionalAction, help="Uses a network trained on the mandatory gestures.")
    group.add_argument(
        "--opt",
        action=argparse.BooleanOptionalAction,
        help="Uses a network trained on the mandatory gestures and additionally some optional gestures. See the README.md for more information.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    """
    Argument validation.

    :param args: The arguments to be validated.
    :return: None
    :raise AttributeError: When test mode is used and no src or dest is specified.
    """
    if args.test:
        if not args.src:
            raise AttributeError("A source csv is needed when using the test mode!")
        if not args.dest:
            raise AttributeError("A dest csv is needed when using the test mode!")


if __name__ == "__main__":
    args = get_args()
    validate_args(args)
    data_src = 0 if args.live else str(args.src)
    nn_src = Paths.MODELS / "network.pkl" if args.mand else Paths.MODELS / "network_opt.pkl"
    pca_src = Paths.MODELS / "pca.pkl" if args.mand else Paths.MODELS / "pca_opt.pkl"
    nn = NeuralNetwork.load(nn_src)

    if args.live:
        print("Starting server...")
        server = multiprocessing.Process(target=slideshow_server.main)
        server.start()

    if args.mand:
        classes = class_names
        window_size = 12
        class_thresholds = {"swipe_left": 0.3, "swipe_right": 0.4, "rotate": 0.6}
        idle_threshold = 1.0
    else:
        classes = class_names_opt
        window_size = 8
        class_thresholds = {
            "swipe_left": 0.3,
            "swipe_right": 0.3,
            "rotate": 0.8,
            "rotate_left": 0.7,
            "flip_table": 0.4,
            "pinch": 0.6,
            "point": 0.6,
            "spread": 0.8,
        }
        idle_threshold = 0.8

    print("Initializing processors.")
    preproc = Preprocessor(pca_path=pca_src, to_delta=features, to_drop=coords_features)
    postproc = Postprocessor(window_size=window_size, idle_thresh=idle_threshold, classes=classes, thresholds=class_thresholds)

    src = args.src if args.test else None
    dest = args.dest if args.test else None

    queue = multiprocessing.Queue(maxsize=1000)

    producer = multiprocessing.Process(
        target=dp.process,
        args=(
            data_src,
            queue,
            args.live,
        ),
    )
    consumer = multiprocessing.Process(
        target=process,
        args=(
            queue,
            nn,
            preproc,
            postproc,
            src,
            dest,
        ),
    )

    print("Starting producer...")
    producer.start()
    print("Starting consumer...")
    consumer.start()
    producer.join()
    consumer.join()
    if args.live:
        server.join()
