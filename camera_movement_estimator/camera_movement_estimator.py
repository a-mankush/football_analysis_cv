import os
import pickle

import cv2
import numpy as np

from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame) -> None:
        """
        Initialize the CameraMovementEstimator class.

        Args:
            frame: The first frame of the video.
        """

        # Set the minimum distance threshold for considering a camera movement.
        self.minimum_distance = 5

        # Convert the first frame to grayscale.
        first_frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask for the features to track.
        # The mask is a binary image where 1 indicates the feature is trackable.
        # In this case, we are setting the first 20 columns and the pixel at (900, 1050) to 1.
        mask_features = np.zeros_like(first_frame_greyscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        # Set the parameters for the goodFeaturesToTrack function.
        # These parameters specify the maximum number of corners, quality level, minimum distance,
        # block size, and the mask for the features to track.
        self.features = dict(
            maxCorners=100,  # Maximum number of corners to detect.
            qualityLevel=0.3,  # Minimum quality level for corner detection.
            minDistance=3,  # Minimum distance between corners.
            blockSize=7,  # Size of the block for corner detection.
            mask=mask_features,  # Mask for the features to track.
        )

        # Set the parameters for the calcOpticalFlowPyrLK function.
        # These parameters specify the window size, maximum level, termination criteria, and the mask for the features to track.
        self.lk_params = dict(
            winSize=(15, 15),  # Window size for Lucas-Kanade optical flow.
            maxLevel=2,  # Maximum level of pyramid for Lucas-Kanade optical flow.
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),  # Termination criteria for Lucas-Kanade optical flow.
        )

    def adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for objects, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    adjusted_position = [
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    ]
                    tracks[objects][frame_num][track_id][
                        "adjusted_position"
                    ] = adjusted_position

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # If we should read from a stub file and the stub path is provided and the file exists,
        # then read the camera movement from the stub file and return it.
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                # Load the camera movement from the stub file.
                camera_movement = pickle.load(f)
                return camera_movement

        # Initialize an empty list to hold the camera movement for each frame.
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and find the features to track.
        old_grey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(image=old_grey, **self.features)

        # Iterate over each frame after the first frame.
        for frame_num in range(1, len(frames)):
            # Convert the current frame to grayscale.
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Use the Lucas-Kanade algorithm to track the features between the current and previous frames.
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                prevImg=old_grey,
                nextImg=frame_gray,
                prevPts=old_features,
                nextPts=None,
                **self.lk_params,
            )

            # Initialize variables to hold the maximum distance and the camera movement for the frame.
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Iterate over each new feature and old feature pair.
            for new, old in zip(new_features, old_features):
                # Convert the new and old features to a single dimension array.
                new_features_points = new.ravel()
                old_features_points = old.ravel()

                # Calculate the distance between the new and old features.
                distance = measure_distance(new_features_points, old_features_points)

                # If the distance is greater than the maximum distance, update the maximum distance and the camera movement for the frame.
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_points, new_features_points
                    )

            # If the maximum distance is greater than the minimum distance, update the camera movement for the frame.
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]

                # Update the old features for the next frame.
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update the old grayscale frame for the next frame.
            old_grey = frame_gray.copy()

        # If a stub path is provided, write the camera movement to the stub file.
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                # Write the camera movement to the stub file.
                pickle.dump(camera_movement, f)

        # Return the camera movement for each frame.
        return camera_movement
