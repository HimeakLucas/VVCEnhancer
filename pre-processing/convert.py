import os
import subprocess

input_dir = 'sequences'  
encoded_dir = 'out_sequences/encoded'  
decoded_dir = 'out_sequences/decoded'  

os.makedirs(encoded_dir, exist_ok=True)
os.makedirs(decoded_dir, exist_ok=True)

vvenc_cmd = "vvencapp"  
vvdec_cmd = "vvdecapp" 

def process_videos():

    num_errors_encoded = 0
    num_errors_decoded = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.y4m'):
            input_path = os.path.join(input_dir, filename)

            encoded_path = os.path.join(encoded_dir, filename.replace('.y4m', '.266'))
            decoded_path = os.path.join(decoded_dir, filename)
            
            try:
                print(f"Coding {filename} to H.266...")
                subprocess.run([vvenc_cmd, '-i', input_path, '-o', encoded_path], check=True)
                print(f"Encoded file saved to {encoded_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error encoding {filename}: {e}")
                num_errors_encoded += 1
                continue
            
            try:
                print(f"Decoding {encoded_path} back to Y4M...")
                subprocess.run([vvdec_cmd, '-b', encoded_path, '-o', decoded_path], check=True)
                print(f"Decoded file saved to {decoded_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error decoding {encoded_path}: {e}")
                num_errors_decoded += 1

    print(f'\nTotal videos failed to encode: {num_errors_encoded}')
    print(f'Total videos that failed to decode: {num_errors_decoded}')

if __name__ == "__main__":
    process_videos()
