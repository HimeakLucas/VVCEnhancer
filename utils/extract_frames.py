import os
import subprocess

def extract_frames_from_video(
    input_path,
    output_video_dir,
    pix_fmt="gray",
    frame_prefix="frame"
):
    os.makedirs(output_video_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-f', 'image2',
        '-pix_fmt', pix_fmt,
        '-vsync', '0',
        os.path.join(output_video_dir, f'{frame_prefix}_%04d.png')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        num_frames = len(os.listdir(output_video_dir))
        print(f'[OK] Extracted {num_frames} frames from {os.path.basename(input_path)}')

    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Error during frame extraction: {error}") from error
