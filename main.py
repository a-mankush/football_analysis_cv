import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssiginer
from team_assignier import TeamAssiginer
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer


def main():
    # Read video
    print("reading Video")
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize the tracker
    print("tracking players")
    tracker = Tracker(model_path="models/best.pt")
    tracks = tracker.get_object_tracks(
        frames=video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Get objects position
    tracker.add_position_to_tracks(tracks)

    # camara movement estimator
    cm_estimator = CameraMovementEstimator(frame=video_frames[0])
    camara_movement_per_frame = cm_estimator.get_camera_movement(
        video_frames, True, "stubs/camera_movement.pkl"
    )

    cm_estimator.adjust_positions_to_tracks(tracks, camara_movement_per_frame)

    # View transformer
    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)
    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the team assignier
    print("assigning teams")
    team_assignier = TeamAssiginer()
    team_assignier.assign_team_color(
        frame=video_frames[0],
        player_detections=tracks["players"][0],
    )

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, player_bbox in player_track.items():
            bbox = player_bbox["bbox"]
            team_id = team_assignier.get_player_team(
                video_frames[frame_num], bbox, player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["color"] = (
                team_assignier.team_colors[team_id]
            )

    # Assign ball Aquisition
    print("assigning ball aquisition")
    player_assigner = PlayerBallAssiginer()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_box = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw annotations
    print("drawing annotations")
    output_video_frames = tracker.draw_annotations(
        video_frames, team_ball_control, tracks, camara_movement_per_frame
    )

    # Save video
    save_video(output_video_frames, "output_videos/output_video_camera_movment.mp4")


if __name__ == "__main__":
    main()
