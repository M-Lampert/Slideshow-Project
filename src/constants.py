from pathlib import Path

from mediapipe.python.solutions.pose import PoseLandmark

landmarks = {
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": PoseLandmark.LEFT_ELBOW,
    "right_elbow": PoseLandmark.RIGHT_ELBOW,
    "left_wrist": PoseLandmark.LEFT_WRIST,
    "right_wrist": PoseLandmark.RIGHT_WRIST,
    "left_pinky": PoseLandmark.LEFT_PINKY,
    "right_pinky": PoseLandmark.RIGHT_PINKY,
    "left_index": PoseLandmark.LEFT_INDEX,
    "right_index": PoseLandmark.RIGHT_INDEX,
    "left_thumb": PoseLandmark.LEFT_THUMB,
    "right_thumb": PoseLandmark.RIGHT_THUMB,
}

event_dict = {
    "swipe_left": "right",
    "swipe_right": "left",
    "rotate": "rotate",
    "pinch": "zoom_out",
    "spread": "zoom_in",
    "point": "point",
    "swipe_up": "down",
    "swipe_down": "up",
    "rotate_left": "rotate_left",
    "flip_table": "flip_table",
}

features = ["%s_%s" % (joint_name, jdn) for joint_name in landmarks for jdn in ["x", "y", "z", "confidence"]]
coords_features = ["%s_%s" % (joint_name, jdn) for joint_name in landmarks for jdn in ["x", "y", "z"]]
confidence_features = ["%s_confidence" % (joint_name) for joint_name in landmarks]
column_names = ["timestamp"] + features

class_names = ["idle", "swipe_left", "swipe_right", "rotate"]
class_names_opt = ["idle", "swipe_left", "swipe_right", "rotate", "rotate_left", "flip_table", "pinch", "point", "spread", "swipe_down", "swipe_up"]


class Paths:
    ROOT = Path(__file__).parent.parent
    MODELS = ROOT / "saved_models"
    DATA = ROOT / Path("data")
    DATA_CSV = DATA / "csv"
    DATA_ELAN = DATA / "elan"
    DATA_OTHER_MAND_DATA = DATA_CSV / "other_group" / "mandatory"
    DATA_OTHER_OPT_DATA = DATA_CSV / "other_group" / "optional"
    DATA_MAND_VAL_DATA = DATA_CSV / "mandatory_gestures" / "validation"
    DATA_MAND_TRAIN_DATA = DATA_CSV / "mandatory_gestures" / "train"
    DATA_OPT_VAL_DATA = DATA_CSV / "optional_gestures" / "validation"
    DATA_OPT_TRAIN_DATA = DATA_CSV / "optional_gestures" / "train"

    def __init__(self):
        paths = [
            Paths.DATA,
            Paths.DATA_CSV,
            Paths.DATA_ELAN,
            Paths.DATA_MAND_VAL_DATA,
            Paths.DATA_MAND_TRAIN_DATA,
            Paths.DATA_OPT_TRAIN_DATA,
            Paths.DATA_OPT_VAL_DATA,
        ]
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True)
