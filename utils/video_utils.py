import cv2


def read_video(video_path):
    """
    Reads a video file from the given `video_path` and returns a list of frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frames.append(frame)

    return frames


def save_video(output_video_frames, output_video_path):
    """
    Saves the given `output_video_frames` as a video file at the specified `output_video_path`.

    Parameters:
        output_video_frames (List): A list of frames representing the video.
        output_video_path (str): The path where the video file will be saved.

    Returns:
        None

    This function uses the OpenCV library to save the video frames as a video file. It first creates a VideoWriter object
    with the specified output video path, fourcc code, frame rate, and frame size. Then, it iterates over each frame in
    the `output_video_frames` list and writes it to the video file using the `write` method of the VideoWriter object.
    Finally, it releases the VideoWriter object to close the video file.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24.0,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    print("Write frames to the video")
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"The Video is saved at path: {output_video_path}")
