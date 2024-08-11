import cv2
import numpy as np


class ViewTransformer:
    def __init__(self) -> None:
        # The width of the tennis court in pixels
        court_width = 68
        court_lenght = 23.32  # 5.83 * 4

        # Define the vertices of the image pixels that correspond to the corners of the field
        # The vertices are represented as a list of tuples, where each tuple contains the x and y coordinates
        # of a vertex in pixels
        self.pixel_vertices = np.array(
            [[110, 1035], [265, 275], [910, 250], [1640, 915]]
        ).astype(np.float32)

        # Define the vertices of the target perspective, which is the field in a rectangular shape
        # The vertices are represented as a list of tuples, where each tuple contains the x and y coordinates
        # of a vertex in meters
        self.target_vertices = np.array(
            [[0, court_lenght], [0, 0], [court_lenght, 0], [court_lenght, court_width]]
        ).astype(np.float32)

        # Calculate the perspective transformation matrix using the pixel vertices and target vertices
        # This matrix is used to transform the image pixels to the target perspective
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        # Convert the point to an integer tuple, since OpenCV functions expect integer values
        # for pixel coordinates
        pixel_point = (int(point[0]), int(point[1]))

        # Check if the point is inside the target vertices
        # cv2.pointPolygonTest returns a non-negative value if the point is inside the polygon
        is_inside = cv2.pointPolygonTest(self.target_vertices, pixel_point, False) >= 0

        # If the point is not inside the target vertices, return None
        if not is_inside:
            return None

        # Reshape the point to a 2D array with a single row and two columns
        # This is the expected format for the perspective transform function
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        # Perform the perspective transform using the OpenCV function cv2.perspectiveTransform
        # The first argument is the array of points to transform, which is reshaped_point
        # The second argument is the perspective transformation matrix, which is self.perspective_transformer
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )

        # Reshape the transformed point back to a 1D array with two elements
        # This is the expected format for the point representation
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # Iterate over the tracks dictionary
        for objects, object_track in tracks.items():
            # Iterate over the frames in each object track
            for frame_num, track in enumerate(object_track):
                # Iterate over the tracks in each frame
                for track_id, track_info in track.items():
                    # Get the adjusted position of the track
                    position = track_info["adjusted_position"]
                    # Convert the position to a NumPy array
                    position = np.array(position)

                    # Transform the position using the perspective transformer
                    transformer_position = self.transform_point(position)

                    # If the transformed position is not None (i.e., it is inside the target vertices)
                    if transformer_position is not None:
                        # Squeeze the transformed position to remove any unnecessary dimensions
                        # and convert it to a list
                        transformer_position = transformer_position.squeeze().tolist()

                        # Add the transformed position to the track dictionary under the
                        # key "transformed_position"
                        tracks[objects][frame_num][track_id][
                            "transformed_position"
                        ] = transformer_position
