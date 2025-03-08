import os
import time
from utils.encode_decode_vvc import encode_decode_video, subprocess

def main():
    input_dir = "../data/sequences"
    encoded_dir = "../data/out_sequences/encoded"
    decoded_dir = "../data/out_sequences/decoded"
    qp_values = [47]
    preset = "faster"

    start = time.time()

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.y4m')]
    print(f"\nProcessing {len(video_files)} videos...")

    stats = {qp: {'success': 0, 'errors': 0} for qp in qp_values}
    errored_videos = []

    for idx, filename in enumerate(video_files, 1):
        video_name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)

        print(f"\nVideo {idx}/{len(video_files)}: {video_name}")

        for qp in qp_values:
            encoded_path = os.path.join(encoded_dir, f'qp{qp}', f"{video_name}.266")
            decoded_path = os.path.join(decoded_dir, f'qp{qp}', f"{video_name}.y4m")

            os.makedirs(os.path.dirname(encoded_path), exist_ok=True)
            os.makedirs(os.path.dirname(decoded_path), exist_ok=True)

            start_v = time.time()
            try:
                encode_decode_video(
                    input_path,
                    encoded_path,
                    decoded_path,
                    qp,
                    preset=preset
                )
                stats[qp]['success'] += 1
                print(f"QP{qp}: OK ({int(time.time()-start_v)}s)")
            except RuntimeError as e:
                stats[qp]['errors'] += 1
                print(f"QP{qp}: Erro - {e}")
                if video_name not in errored_videos:
                    errored_videos.append(video_name)

    print("\nSummary:")
    for qp in qp_values:
        total = stats[qp]['success'] + stats[qp]['errors']
        print(f"QP{qp}: {stats[qp]['success']} success, {stats[qp]['errors']} errors")

    print("\nVideos with errors:")
    for video_name in errored_videos:
        print(video_name)

    print(f"Total time: ({int(time.time()-start)}s)")

if __name__ == "__main__":
    main()
