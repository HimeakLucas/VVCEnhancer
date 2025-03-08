import os
import subprocess

def encode_decode_video(
    input_path,
    encoded_path,
    decoded_path,
    qp,
    vvcenc_cmd='vvencapp',
    vvdec_cmd='vvdecapp',
    preset='medium'
): 
    try:
        # Encode
        subprocess.run([
            vvcenc_cmd, '-i', input_path, '-o', encoded_path,
            '-q', str(qp), '--preset', preset
        ], check=True, capture_output=True)
        
        # Decode
        subprocess.run([
            vvdec_cmd, '-b', encoded_path, '--y4m', '-o', decoded_path
        ], check=True, capture_output=True)

        return True, None  # Sucesso

    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode().strip()  # Erro
