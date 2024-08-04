from utils import read_video, save_video

def main():
    # Read video
    frames = read_video("input_videos/08fd33_4.mp4")
    save_video(frames, "output_videos/08fd33_4.mp4") 
if __name__ == "__main__":
    main()
    
    