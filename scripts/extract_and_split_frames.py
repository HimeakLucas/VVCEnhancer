import os
from sklearn.model_selection import train_test_split
from utils.extract_frames import extract_frames_from_video

def extract_frames_from_video_list(video_files, input_dir, output_dir, frame_prefix="frame", pix_fmt="gray"):
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        
        os.makedirs(video_output_dir, exist_ok=True)
        try:
            extract_frames_from_video(
                input_path=input_path,
                output_video_dir=video_output_dir,
                frame_prefix=frame_prefix,
                pix_fmt=pix_fmt
            )
        except RuntimeError as e:
            print(f"Error processing {video_name}: {e}")

def main():
    original_input_dir = "../data/sequences"
    processed_input_base = "../data/out_sequences/decoded"
    output_base_dir = "../data/frames_y"
    test_size = 0.2
    random_state = 42


    all_videos = [f for f in os.listdir(original_input_dir) if f.endswith('.y4m')]
    train_videos, val_videos = train_test_split(all_videos, test_size=test_size, random_state=random_state)

    print("\nProcessing original videos:")
    extract_frames_from_video_list(train_videos, original_input_dir, os.path.join(output_base_dir, "train", "original"))
    extract_frames_from_video_list(val_videos, original_input_dir, os.path.join(output_base_dir, "val", "original"))

    #Assume que a estrutura de pastas de vídeos codificados esteja separada por QP.
    #Os mesmos vídeos são extraídos para cada QP, garantindo consistência entre os datasets.
    print("\nProcessing encoded videos:")
    for qp_folder in os.listdir(processed_input_base):
        qp_path = os.path.join(processed_input_base, qp_folder)
        
        if not os.path.isdir(qp_path):
            continue
            
        print(f"\nQP: {qp_folder}")
        # Verifica quais vídeos existem nesta pasta QP
        available_videos = set(os.listdir(qp_path))
        qp_train = [v for v in train_videos if v in available_videos]
        qp_val = [v for v in val_videos if v in available_videos]

        extract_frames_from_video_list(qp_train, qp_path, os.path.join(output_base_dir, "train", qp_folder))
        extract_frames_from_video_list(qp_val, qp_path, os.path.join(output_base_dir, "val", qp_folder))

if __name__ == "__main__":
    main()