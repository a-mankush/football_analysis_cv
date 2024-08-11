import cv2

from utils import get_foot_position, measure_distance


class SpeedAndDistanceEstimator:
    def __init__(self) -> None:
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for objects, object_track in tracks.items():
            if objects == "ball" or objects == "referees":
                continue
            number_of_frames = len(object_track)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_track[frame_num].items():
                    if track_id not in object_track[last_frame]:
                        continue

                    start_position = object_track[frame_num][track_id][
                        "transformed_position"
                    ]
                    end_position = object_track[last_frame][track_id][
                        "transformed_position"
                    ]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    speed_meter_per_second = distance_covered / time_elapsed
                    speed_km_per_hr = speed_meter_per_second * 3.16

                    if objects not in total_distance:
                        total_distance[objects] = {}  # total_distance['players']

                    if track_id not in total_distance[objects]:
                        total_distance[objects][track_id] = 0

                    total_distance[objects][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[objects][frame_num_batch]:
                            continue
                        tracks[objects][frame_num_batch][track_id][
                            "speed"
                        ] = speed_km_per_hr
                        tracks[objects][frame_num_batch][track_id]["distance"] = (
                            total_distance[objects][track_id]
                        )

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for objects, object_track in tracks.items():
                if objects == "ball" or objects == "referees":
                    continue
                for _, player_info in object_track[frame_num].items():
                    if "speed" in player_info:
                        speed = player_info.get("speed", None)
                        distance = player_info.get("distance", None)
                        if speed is None or distance is None:
                            continue

                        bbox = player_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))

                        cv2.putText(
                            frame,
                            f"{speed:.2f} km/h",
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,  # thickness
                        )

                        cv2.putText(
                            frame,
                            f"{distance:.2f} m",
                            (position[0], position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2,  # thickness
                        )
            output_frames.append(frame)

        return output_frames
