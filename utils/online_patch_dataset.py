import os
import random
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class OnlinePatchDataset(Dataset):
    def __init__(self, original_dir, processed_dir, patch_size=64, 
                 patches_per_frame=3, transform=None):

        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.transform = transform
        self.pairs = []

        # Mapeamento de vídeos e frames
        self.original_videos = self._map_video_frames(original_dir)
        self.processed_videos = self._map_video_frames(processed_dir)

        # Valida e cria pares
        self._validate_directories()
        self._create_pairs()

    def _map_video_frames(self, root_dir):
        #Mapeia a estrutura de diretórios para vídeos e seus frames
        #Assume que a estrutura é: root_dir/video_name/frame_XXXX.png
        video_map = {}
        for video_name in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_name)
            if os.path.isdir(video_path):
                frames = []
                for fname in sorted(os.listdir(video_path)):
                    if fname.endswith('.png'):
                        frame_num = self._extract_frame_number(fname)
                        frames.append({
                            'path': os.path.join(video_path, fname),
                            'number': frame_num
                        })
                if frames:
                    video_map[video_name] = frames
        return video_map

    def _extract_frame_number(self, filename):
        base = os.path.splitext(filename)[0]
        return int(base.split('_')[-1])

    def _validate_directories(self):
        #Valida correspondência entre vídeos originais e processados
        keys_to_remove = []
        for processed_video in self.processed_videos:
            if processed_video not in self.original_videos:
                print(f"Warning: Vídeo processado '{processed_video}' sem correspondente. Removendo.")
                keys_to_remove.append(processed_video)
        for key in keys_to_remove:
            del self.processed_videos[key]

    def _create_pairs(self):
        for video_name, proc_frames in self.processed_videos.items():
            original_frames = self.original_videos.get(video_name, [])
            original_frame_map = {f['number']: f['path'] for f in original_frames}
            
            for proc_frame in proc_frames:
                frame_num = proc_frame['number']
                if frame_num in original_frame_map:
                    # Adiciona N entradas para o mesmo par de frames
                    for _ in range(self.patches_per_frame):
                        self.pairs.append((
                            original_frame_map[frame_num],
                            proc_frame['path']
                        ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig_path, proc_path = self.pairs[idx]

        orig_img = Image.open(orig_path).convert('L')
        proc_img = Image.open(proc_path).convert('L')

        # Verificação adicional de tamanho
        if orig_img.size[0] < self.patch_size or orig_img.size[1] < self.patch_size:
            raise ValueError(f"Imagem muito pequena: {orig_path} ({orig_img.size})")

        if orig_img.size != proc_img.size:
            raise ValueError(f"Tamanho diferente: {orig_path} vs {proc_path}")

        # Gera patch único para cada acesso
        patch_box = self._select_patch(np.array(orig_img))
        
        orig_patch = orig_img.crop(patch_box)
        proc_patch = proc_img.crop(patch_box)

        if self.transform:
            orig_patch = self.transform(orig_patch)
            proc_patch = self.transform(proc_patch)

        return orig_patch, proc_patch

    def _select_patch(self, img_np):
        #Seleção híbrida de patches (70% bordas, 30% aleatório)
        h, w = img_np.shape[:2]  
        
        if random.random() < 0.7:
            edges = cv2.Canny(img_np, 50, 150)
            
            for _ in range(50):
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size) 
                edge_patch = edges[y:y+self.patch_size, x:x+self.patch_size]
                if edge_patch.mean() > 3:
                    return (x, y, x+self.patch_size, y+self.patch_size)
        
        # Fallback aleatório
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size) 
        return (x, y, x+self.patch_size, y+self.patch_size)

if __name__ == "__main__":
    # Example: creating traning dataset
    train_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/train/original",
        processed_dir="../data/frames_y/train/qp63",
        patch_size=64,
        patches_per_frame=6,
        transform=transforms.ToTensor()
    )
    print("Number of training pairs:", len(train_dataset))

    # Example: creating validation dataset
    val_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/val/original",
        processed_dir="../data/frames_y/val/qp63",
        patch_size=64,
        patches_per_frame=6,
        transform=transforms.ToTensor()
    )

    print("Number of validation pairs:", len(val_dataset))