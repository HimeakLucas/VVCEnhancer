import os
import numpy as np
import cv2

def calculate_psnr(original_frame, restored_frame):

    original_frame = original_frame.astype(np.float32)
    restored_frame = restored_frame.astype(np.float32)
    
    mse = np.mean((original_frame - restored_frame) ** 2)
    
    if mse == 0:
        return float('inf')
    
    pixel_max = 255.0 
    psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    
    return psnr


def calculate_video_psnr(original_frames_dir, restored_frames_dir):

    def load_frame_as_grayscale(frame_path):
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        return frame

    original_frames = sorted(os.listdir(original_frames_dir))
    restored_frames = sorted(os.listdir(restored_frames_dir))
    
    psnr_values = []
    
    if len(original_frames) != len(restored_frames):
        raise ValueError("The videos don't have the same amout of frames")
    
    for original_frame_name, restored_frame_name in zip(original_frames, restored_frames):

        original_frame_path = os.path.join(original_frames_dir, original_frame_name)
        restored_frame_path = os.path.join(restored_frames_dir, restored_frame_name)
        
        original_frame = load_frame_as_grayscale(original_frame_path)
        restored_frame = load_frame_as_grayscale(restored_frame_path)

        psnr = calculate_psnr(original_frame, restored_frame)
        psnr_values.append(psnr)

    if psnr_values:
        average_psnr = np.mean(psnr_values)
        return average_psnr
    else:
        return None 


if __name__ == "__main__":

    original_frames_dir = "sequences/video1_frames"
    restored_frames_dir = "out_sequences/fast_preset/video1_restored_frames"

    average_psnr = calculate_video_psnr(original_frames_dir, restored_frames_dir)

    if average_psnr is not None:
        print(f"Mean PSNR: {average_psnr:.2f} dB")
    else:
        print("Error")
