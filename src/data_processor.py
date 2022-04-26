import datetime as dt
import multiprocessing
from pathlib import Path
from typing import List, Union

import cv2
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose
import pandas as pd

import src.constants as const

available_transforms = {"vertical_flip": 0, "horizontal_flip": 1}


def process(
    source: Union[str, int],
    queue: multiprocessing.Queue = None,
    show_video: bool = False,
    to_csv: Path = None,
    path_to_elan: Path = None,
    transforms: List[str] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """
    Processes either a given video given by a path or live image from a webcam given by index.

    :param source: Either int or str. If video is to be processed from a webcam, then give the index of your webcam. If a video file is to be processed, give a
    path.
    :param path_to_elan: If given, merge ELAN annotations with the created csv.
    :param queue:
    :param show_video: bool if a video output is to be shown while processing.
    :param to_csv: Path to save the created .csv-file to. If None, no .csv-file will be created.
    :param transforms: List of transformations to be done on the video. Defaults to None.
    :param min_detection_confidence:
    :param min_tracking_confidence:
    :return:
    """
    if transforms is None:
        transforms = []
    if min_detection_confidence is None:
        min_detection_confidence = 0.5
    if min_tracking_confidence is None:
        min_tracking_confidence = 0.5
    start_time = dt.datetime.now()
    writer = CSVDataWriter()
    if isinstance(source, int) or source.endswith(".mov") or source.endswith(".mp4"):
        cap = cv2.VideoCapture(source)
        success = True
        with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
            while cap.isOpened() and success:
                success, image = cap.read()
                if not success:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                for transform in transforms:
                    image = cv2.flip(image, available_transforms[transform])
                results = pose.process(image)
                pose_landmarks = results.pose_landmarks

                if show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    mp_drawing.draw_landmarks(
                        image, pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # image = cv2.resize(image, (1280, 720))
                    cv2.imshow("MediaPipe Pose", image)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                if pose_landmarks:
                    if type(source) is int:
                        timestamp = (dt.datetime.now() - start_time).total_seconds() * 1000  # Get the difference in milliseconds
                    else:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                    if queue:
                        queue.put(to_df(timestamp, pose_landmarks))
                    if to_csv:
                        writer.read_data(to_df(timestamp, pose_landmarks))
        cap.release()
    elif source.endswith(".csv"):
        csv = pd.read_csv(source).loc[:, const.column_names]
        for i in range(len(csv)):
            row = csv.iloc[i, :]
            row_df = row.to_frame()
            row_df = row_df.T
            if queue:
                queue.put(row_df)
            if to_csv:
                writer.read_data(row_df)
    else:
        raise AttributeError(f"Unknown file format {source.split('.')[-1]}!")

    if queue:
        queue.put(None)  # Close message
    if to_csv:
        writer.to_csv(to_csv, path_to_elan, transforms)


def merge_csv_elan(frames: pd.DataFrame, path_to_elan: Path):
    """
    Adds the ground truth to the given dataframe by reading the elan-file.

    :param frames: Pandas Dataframe containing the video-transcript without ground truth
    :param path_to_elan: Corresponding ground truth in elan format
    :return: Pandas Dataframe with ground truth column
    """
    frames["ground_truth"] = "idle"

    elan = pd.read_csv(path_to_elan, sep="\t", header=None, usecols=[3, 5, 8], names=["start", "end", "label"])
    elan["start"] = pd.to_timedelta(elan["start"], unit="s")
    elan["end"] = pd.to_timedelta(elan["end"], unit="s")

    for idx, annotation in elan.iterrows():
        annotated_frames = (frames.index >= annotation["start"]) & (frames.index <= annotation["end"])
        frames.loc[annotated_frames, "ground_truth"] = annotation["label"]

    return frames


def to_df(timestamp, pose_landmarks):
    """
    Turns the mediapipe representation into a pandas dataframe.

    :param timestamp: The timestamp of the current frame.
    :param pose_landmarks: The pose landmarks as given by mediapipe
    :return: Pandas dataframe with one row (the given frame).
    """
    result = {"timestamp": [timestamp]}
    for name, landmark in const.landmarks.items():
        result[f"{name}_x"] = [pose_landmarks.landmark[landmark].x]
        result[f"{name}_y"] = [pose_landmarks.landmark[landmark].y]
        result[f"{name}_z"] = [pose_landmarks.landmark[landmark].z]
        result[f"{name}_confidence"] = [pose_landmarks.landmark[landmark].visibility]
    return pd.DataFrame(result)


class CSVDataWriter:
    """
    Helper class to create a .csv-file from a dataframe or a stream of many dataframes with one row (creating video transcript).
    """

    def __init__(self, data: pd.DataFrame = None):
        if data is None:
            self.frames = pd.DataFrame(columns=const.column_names)
        else:
            self.frames = data

    def read_data(self, data: pd.DataFrame):
        """
        Adds the given frame as row to the dataframe in which all frames are saved.

        :param data: The frame that should be added.
        :return: None
        """
        self.frames = pd.concat([self.frames, data])

    def to_csv(self, output_path: Union[Path, str], path_to_elan: Union[Path, str] = None, transforms=None):
        """
        Writes the saved frames to .csv.

        :param output_path: The path to save the frames in.
        :param path_to_elan: The path to the ground truth. Default is none, then no ground truth will be added.
        :param transforms: Transforms the labels if the data was augmented.
        :return: None
        """
        if transforms is None:
            transforms = []
        self.frames["timestamp"] = pd.to_timedelta(self.frames["timestamp"], unit="ms")
        self.frames = self.frames.set_index("timestamp")

        if path_to_elan:
            self.frames = merge_csv_elan(self.frames, path_to_elan)

        for transform in transforms:
            if available_transforms[transform] == 0:
                self.frames["ground_truth"] = self.frames["ground_truth"].replace(to_replace=["swipe_up", "swipe_down"], value=["swipe_down", "swipe_up"])
            elif available_transforms[transform] == 1:
                self.frames["ground_truth"] = self.frames["ground_truth"].replace(to_replace=["swipe_left", "swipe_right"], value=["swipe_right", "swipe_left"])
        self.frames.round(20).to_csv(output_path)


def video_to_csv(
    dest_path: Union[str, Path], video_path: Union[str, Path], txt_path: Union[str, Path] = None, show_video: bool = False, transformations: List[str] = None
):
    """
    Creates a .csv-file from a given video and its elan annotation. The data will be saved to data/csv/time_of_recording.csv

    :param dest_path: The path to save the .csv-files to.
    :param video_path: The source path of the video.
    :param txt_path: The source path of the elan annotation. If None, the .csv-file will not contain a ground truth. Default: None
    :param show_video: If the video should be shown during conversion. Default: False
    :param transformations: List of transformations to be done on the video. Default: None.
    :return: None
    """
    print(f"Creating csv for {video_path}. It will be saved to {dest_path}")
    producer = multiprocessing.Process(
        target=process,
        args=(
            video_path,
            None,
            show_video,
            dest_path,
            txt_path,
            transformations,
        ),
    )
    producer.start()
    producer.join()
