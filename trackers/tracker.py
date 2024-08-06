import supervision as sv
from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += batch_detections
            break
        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        for frame_num, detection in enumerate(detections):
            cls_name = detection.names  # {0: 'person', 1: 'car', ....}
            cls_name_inv = {
                v: k for k, v in cls_name.items()
            }  # {'person': 0, 'car': 1, ....}

            # Convert the detections from ultralytics to Supervision format
            supervision_detections = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper to player object
            for object_ind, class_id in enumerate(supervision_detections.class_id):
                if cls_name[class_id] == "goalkeeper":
                    supervision_detections.class_id[object_ind] = cls_name_inv["player"]
            print(supervision_detections)
            break
