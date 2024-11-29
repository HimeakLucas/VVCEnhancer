import os
import subprocess

def extract_y_channel_frames(y4m_video_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    command = [
        "ffmpeg",
        "-y",
        "-i", y4m_video_path,
        "-vf", "extractplanes=y",
        os.path.join(output_dir, "frame_%04d.pgm")
    ]
    
    subprocess.run(command, check=True)

if __name__ == "__main__":
    video_path = "sequences/crowd_run_1080p50_60f.y4m" 
    output_dir = "frames/video1/"
    extract_y_channel_frames(video_path, output_dir)