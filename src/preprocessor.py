from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from framework.pca import PCA

from src.constants import coords_features


class Preprocessor:
    def __init__(self, pca_path: Union[str, Path] = None, to_delta: List[str] = coords_features, to_drop: List[str] = coords_features):
        """
        Initializes the preprocessor.

        :param pca_path: Path to saved pca instance. If a PCA-path is given, the features can be transformed with the corresponding projection matrix.
        :param to_delta: All columns that should be used to create a new feature that contains the difference of this column to the last frame.
        Default: constants.coords_features
        :param to_drop: All columns that should be dropped. Default: constants.coords_features
        """

        self.last_frame = None
        if pca_path:
            self.pca: Optional[PCA] = PCA.load(pca_path)
        else:
            self.pca: Optional[PCA] = None
        self.to_delta: List[str] = to_delta
        self.to_drop: List[str] = ["timestamp"] + to_drop

    def preprocess(self, curr_frame: Union[List[pd.DataFrame], pd.DataFrame], batch=False, pca: bool = False):
        """
        Complete preprocessing pipeline.

        :param curr_frame: The current frame that should be preprocessed as pandas DataFrame. If batch is True, it should contain all frames as a list of
        DataFrames (one for each video) or a single DataFrame.
        :param batch: If the preprocessing pipeline should work in batch-mode or not. Batch-mode only works with ground_truth.
        :param pca: If pca should be used to transform the training data or not. In batch-mode, a corresponding projection matrix can be initialized
        automatically from the data. In non-batch-mode, a pca has to exist.
        """
        if batch:
            # Since we are subtracting two frames, we cannot append different videos before preprocessing,
            # because this would create false data between the videos.
            # So if more than one video is used, they need to be preprocessed separately at first
            videos = [curr_frame] if isinstance(curr_frame, pd.DataFrame) else curr_frame

            x_s = None
            y_s = None

            for frames in videos:
                diffs = self.calc_differences_batch(frames[self.to_delta])
                frames = pd.concat([frames.drop(self.to_drop, axis=1).iloc[1:], diffs.iloc[1:]], axis=1)
                if x_s is None and y_s is None:
                    x_s = frames.drop("ground_truth", axis=1).to_numpy(dtype=np.float64)
                    y_s = frames["ground_truth"].to_numpy()
                else:
                    x_s = np.append(x_s, frames.drop("ground_truth", axis=1).to_numpy(dtype=np.float64), axis=0)
                    y_s = np.append(y_s, frames["ground_truth"].to_numpy(), axis=0)

            # PCA transformation
            if pca:
                if self.pca is None:
                    self.pca = PCA.create_PCA(data=x_s, dimensions_or_variance=0.99)

                if isinstance(self.pca, PCA):
                    x_s = self.pca.transform(x_s)

            return x_s, y_s

        else:
            if self.last_frame is not None:
                diffs = self.calc_differences(self.last_frame[self.to_delta], curr_frame[self.to_delta])
                self.last_frame = curr_frame
                curr_frame = pd.concat([curr_frame.drop(self.to_drop, axis=1), diffs], axis=1)
                x = curr_frame.to_numpy(dtype=np.float64)
                if pca:
                    if self.pca is not None and self.pca.projection_matrix is not None:
                        x = self.pca.transform(x)
                    else:
                        raise ValueError("The pca is not initialized.")
                return x

            else:
                self.last_frame = curr_frame
                return None

    def calc_differences(self, last_frame: pd.DataFrame, curr_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to calculate the difference for two given frames.

        :param last_frame: The columns of the last frame for which the differences should be calculated.
        :param curr_frame: The columns of the current frame for which the differences should be calculated.
        :return: Pandas Dataframe with the computed differences. The column names changed to "..._delta_..."
        """
        result = curr_frame - last_frame.to_numpy()
        result.columns = result.columns.str.replace("_", "_delta_")
        return result

    def calc_differences_batch(self, in_df: pd.DataFrame):
        """
        Helper method to calculate the difference for a batch of frames.

        :param in_df: Input dataframe for which the differences should be calculated for.
        :return: Pandas Dataframe with the computed differences. The column names changed to "..._delta_..."
        """
        result = in_df.diff()
        result.columns = result.columns.str.replace("_", "_delta_")
        return result

    def save_pca(self, path: Union[str, Path]):
        """
        Method to save the current pca-object.

        :param path: Where to save to.
        :return: None
        """
        if self.pca:
            self.pca.save(path)
