from collections import deque
from typing import List, Tuple, Union

import numpy as np
from framework.utils import one_hot_vector_to_classes

from src.constants import class_names


class Postprocessor:
    """
    Handles the postprocessing. The handler gets one prediction and emits one event.
    """

    def __init__(self, window_size=10, idle_thresh=1, classes: List[str] = class_names, thresholds=None):
        if thresholds is None:
            thresholds = {"rotate": 0.5, "swipe_left": 0.5, "swipe_right": 0.5}
        self.window = deque(["idle"] * window_size, maxlen=window_size)
        self.window_size = window_size
        self.idle_thresh = idle_thresh
        self.class_names = classes
        self.thresholds = thresholds
        self.last_event = "idle"

    def postprocess(self, pred: np.ndarray, batch: bool = False) -> Union[str, List[str]]:
        """
        Complete postprocessing pipeline

        :param pred: The prediction for the current frame as probability vector.
        :param batch: If a whole batch of predictions should be postprocessed at once. Used for training and evaluation.
        :return: Event or list of events as string.
        """
        if batch:
            pred = one_hot_vector_to_classes(pred, self.class_names, sparse=False)
            return self.get_events(pred)
        else:
            pred = self.class_names[np.argmax(pred)]
            return self.get_event(pred)

    def get_event(self, curr_pred: str) -> str:
        """
        Gets one prediction and returns one event via sliding window approach.

        :param curr_pred: The current frame
        :return The predicted event
        """
        self.window.append(curr_pred)

        idle_conf = self.window.count("idle") / self.window_size

        if idle_conf >= self.idle_thresh:
            self.last_event = "idle"
            res_event = "idle"
        elif (max_tuple := self.get_max_confidence())[0] >= self.thresholds[max_tuple[1]] and self.last_event == "idle":
            res_event = max_tuple[1]
        else:
            res_event = "idle"

        if res_event == self.last_event:
            res_event = "idle"
        elif res_event != "idle":
            self.last_event = res_event

        return res_event

    def get_events(self, predictions: np.ndarray) -> List[str]:
        """
        Batch mode for get_event.

        :param predictions: List of predictions
        :return: List of events
        """
        return [self.get_event(prediction) for prediction in predictions]

    def get_max_confidence(self) -> Tuple[float, str]:
        """
        Helper function: Finds the event that is most often in the current window and returns its confidence as well as the class itself.

        :return: Confidence and event of the max in window.
        """
        curr_max = 0
        curr_max_class = ""
        for curr_class in self.thresholds.keys():
            curr_count = self.window.count(curr_class)
            if curr_count > curr_max:
                curr_max_class = curr_class
                curr_max = curr_count
        return curr_max / self.window_size, curr_max_class
