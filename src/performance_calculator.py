# This is copied from the getting started repository with some small modifications


from typing import List, Union

import numpy as np
import pandas as pd


def count_individual_gestures(ground_truth):
    return (ground_truth.iloc[:-1] != ground_truth.shift(-1).iloc[:-1]).sum() // 2


def calculate_scores(
    events: np.ndarray, ground_truth: np.ndarray, bonus: float = 10, malus: float = 0.2, verbose: bool = True, gestures: Union[str, List[str]] = "all"
):
    """
    :param events: Array of emitted events
    :param ground_truth: Array of ground truth
    :param bonus: Bonus in percent to be added in the end
    :param malus: Malus to be subtracted for each mistake
    :param verbose: Index of frames with malus, marks-count and performance score are printed, iff true
    :param gestures: Gestures to be considered in the calculation, "all" by default
    :return: Score in the range [0, 100 + bonus]
    """

    events = pd.Series(events)
    ground_truth = pd.Series(ground_truth)

    if gestures != "all":
        if isinstance(gestures, str):
            gestures = [gestures]
        gesture_filter = np.zeros_like(ground_truth)
        for g in gestures + ["idle"]:
            gesture_filter = gesture_filter | (ground_truth == g)
        events = events[gesture_filter]
        ground_truth = ground_truth[gesture_filter]

    assert len(events) == len(ground_truth), "Error: the CSV files differ in length!"

    num_frames = len(ground_truth)
    num_total_gestures = count_individual_gestures(ground_truth)

    last_frame_gesture = "idle"
    current_gesture_detected = False

    marks = 0

    for frame_idx in range(num_frames):
        current_gesture = ground_truth.iloc[frame_idx]
        current_event = events.iloc[frame_idx]

        event_fired = current_event != "idle"
        gesture_in_progress = current_gesture != "idle"

        fired_wrong_event = gesture_in_progress and event_fired and current_event != current_gesture

        fired_event_but_no_gesture_in_progress = event_fired and not gesture_in_progress
        fired_event_more_than_once = event_fired and current_gesture_detected

        if last_frame_gesture != current_gesture:
            gesture_ended_but_no_event_fired = not gesture_in_progress and not current_gesture_detected
            current_gesture_detected = False
        else:
            gesture_ended_but_no_event_fired = False

        if fired_event_but_no_gesture_in_progress or fired_event_more_than_once or fired_wrong_event or gesture_ended_but_no_event_fired:
            marks -= malus
            if verbose:
                print(f"Malus in frame: {frame_idx}")
        elif gesture_in_progress and current_event == current_gesture:
            marks += 1
            current_gesture_detected = True

        last_frame_gesture = current_gesture

    total_points = num_total_gestures
    score = max((marks / total_points) * 100 + bonus, 0)

    if verbose:
        print("marks: %d/%d" % (marks, total_points))
    if verbose:
        print("performance score: %.2f%%" % score)

    return score
