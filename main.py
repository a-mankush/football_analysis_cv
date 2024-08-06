from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize the tracker
    tracker = Tracker(model_path="models/best.pt")
    tracks = tracker.get_object_tracks(
        frames=video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )
    print(f"==>> tracks: {tracks}")

    # Save video
    # save_video(video_frames, "output_videos/08fd33_4.mp4")


if __name__ == "__main__":
    main()
