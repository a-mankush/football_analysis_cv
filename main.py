import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssiginer
from team_assignier import TeamAssiginer
from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read the video file
    print("reading Video")
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize the tracker for the players in the video
    print("tracking players")
    tracker = Tracker(model_path="models/best.pt")
    
    # Get the object tracks for the players in the video
    tracks = tracker.get_object_tracks(
        frames=video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Estimate the camera movement for each frame in the video
    print("estimating camera movement")
    # camara movement estimator
    cm_estimator = CameraMovementEstimator(frame=video_frames[0])
    camara_movement_per_frame = cm_estimator.get_camera_movement(
        video_frames, True, "stubs/camera_movement.pkl"
    )

    # Interpolate the ball positions for each frame in the video
    print("interpolating ball positions")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the team assignier for assigning teams to players
    print("assigning teams")
    team_assignier = TeamAssiginer()
    
    # Assign colors to each team based on the first frame of the video
    team_assignier.assign_team_color(
        frame=video_frames[0],
        player_detections=tracks["players"][0],
    )

    # For each frame in the video, assign a team to each player
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, player_bbox in player_track.items():
            bbox = player_bbox["bbox"]
            # Assign a team to the player based on their bounding box in the current frame
            team_id = team_assignier.get_player_team(
                video_frames[frame_num], bbox, player_id
            )
            # Store the team id and color for the player in the tracks dictionary
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["color"] = (
                team_assignier.team_colors[team_id]
            )

    # Initialize the player ball assigner for assigning ball possession to players
    print("assigning ball aquisition")
    player_assigner = PlayerBallAssiginer()
    
    # Initialize a list to store the team with ball possession for each frame
    team_ball_control = []
    
    # For each frame in the video, assign ball possession to a player
    for frame_num, player_track in enumerate(tracks["players"]):
        # Get the bounding box of the ball in the current frame
        ball_box = tracks["ball"][frame_num][1]["bbox"]
        # Assign the ball to a player based on their bounding box in the current frame
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)

        # If a player was assigned the ball, mark them as having the ball and store the team with ball possession
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        # If no player was assigned the ball, keep the team with ball possession the same as the previous frame
        else:
            team_ball_control.append(team_ball_control[-1])
    
    # Convert the team ball control list to a numpy array
    team_ball_control = np.array(team_ball_control)

    # Draw annotations for each frame in the video
    print("drawing annotations")
    output_video_frames = tracker.draw_annotations(
        video_frames, team_ball_control, tracks, camara_movement_per_frame
    )

    # Save the annotated video to a file
    save_video(output_video_frames, "output_videos/output_video_camera_movment.mp4")


if __name__ == "__main__":
    main()
