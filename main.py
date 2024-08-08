import cv2

from team_assignier import TeamAssiginer
from trackers import Tracker
from utils import read_video, save_video


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

    # Draw annotations
    print("drawing annotations")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video_2.mp4")


if __name__ == "__main__":
    main()
