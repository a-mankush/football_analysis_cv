import os
import pickle

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from sklearn.cluster import KMeans
from ultralytics import YOLO

from utils import get_bbox_width, get_center_of_bbox


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

        return detections  # List[dict] {'boxes': [[],[],[]], 'conf': [], 'cls_names': [0,0,0,2,2,1,1,1,1]}

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

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

            # Tracker objects
            detection_with_tracker = self.tracker.update_with_detections(
                supervision_detections
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracker:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_name_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in supervision_detections:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            if stub_path is not None:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)

        return tracks

    def draw_triangel(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox=bbox)

        traiangel_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [traiangel_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traiangel_points], 0, (0, 0, 0), 2)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # want to place ellipse at the bottom
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Draw Rectangel
        rectangle_width = 40
        rectangle_height = 20
        rectangle_x1 = x_center - rectangle_width // 2
        rectangle_x2 = x_center + rectangle_width // 2
        rectangle_y1 = (y2 - rectangle_height // 2) + 15
        rectangle_y2 = (y2 + rectangle_height // 2) + 15
        if track_id is not None:
            cv2.rectangle(
                frame,
                pt1=(rectangle_x1, rectangle_y1),
                pt2=(rectangle_x2, rectangle_y2),
                color=color,
                thickness=cv2.FILLED,
            )
            text_x1 = rectangle_x1 + 12
            if track_id > 99:
                text_x1 -= 10

            cv2.putText(
                frame,
                str(track_id),
                (text_x1, rectangle_y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black
                2,
            )

        return frame

    def draw_annotations(self, video_frames, tracks=None):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("color", (0, 0, 225))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangel(frame, ball["bbox"], (0, 255, 0))  # b.g.r

            output_video_frames.append(frame)
        return output_video_frames

    def interpolate_ball_positions(self, ball_positionss):
        ball_positionss = [
            frame.get(1, {}).get("bbox", []) for frame in ball_positionss
        ]
        df_ball_positions = pd.DataFrame(
            ball_positionss, columns=["x1", "y1", "x2", "y2"]
        )

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        return [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
