import os
import random
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms.functional import rotate, hflip, vflip
import random


class OnlinePatchDataset(Dataset):
    def __init__(self, original_dir, processed_dir, patch_size=64, patches_per_frame=3, 
                 transform=None, use_sliding_window=False, stride=None, edge_prob=0.7):

        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.transform = transform
        self.edge_prob = edge_prob
        self.use_sliding_window = use_sliding_window
        if self.use_sliding_window:
            self.stride = stride if stride is not None else patch_size // 2
        self.pairs = []

        self.original_videos = self._map_video_frames(original_dir)
        self.processed_videos = self._map_video_frames(processed_dir)

        self._validate_directories()
        
        if self.use_sliding_window:
            self._create_pairs_sliding_window()
            print(f"Created {len(self.pairs)} pairs using sliding window.")
        else:
            self._create_pairs()
            print(f"Created {len(self.pairs)} pairs using hybrid mode.")

    def _map_video_frames(self, root_dir):
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
        keys_to_remove = []
        for processed_video in self.processed_videos:
            if processed_video not in self.original_videos:
                print(f"Processed video '{processed_video}' has no match. Removing.")
                keys_to_remove.append(processed_video)
        for key in keys_to_remove:
            del self.processed_videos[key]

    def _create_pairs(self):
        # Modo hibrido: para cada frame processado, gerar patches patches_per_frame vezes
        for video_name, proc_frames in self.processed_videos.items():
            original_frames = self.original_videos.get(video_name, [])
            original_frame_map = {f['number']: f['path'] for f in original_frames}
            
            for proc_frame in proc_frames:
                frame_num = proc_frame['number']
                if frame_num in original_frame_map:
                    for _ in range(self.patches_per_frame):
                        self.pairs.append((
                            original_frame_map[frame_num],
                            proc_frame['path']
                        ))

    def _create_pairs_sliding_window(self):
        # Sliding window: para cada frame processado, gerar patches com stride
        for video_name, proc_frames in self.processed_videos.items():
            original_frames = self.original_videos.get(video_name, [])
            original_frame_map = {f['number']: f['path'] for f in original_frames}

            for proc_frame in proc_frames:
                frame_num = proc_frame['number']
                if frame_num in original_frame_map:
                    orig_path = original_frame_map[frame_num]
                    proc_path = proc_frame['path']

                    orig_img = Image.open(orig_path).convert('L')
                    w, h = orig_img.size
                    
                    # Faz offsets aleatórios para gerar patches
                    x_offset = random.randint(0, self.stride) if self.stride > 0 else 0
                    y_offset = random.randint(0, self.stride) if self.stride > 0 else 0
                    
                    # Generate patches with sliding window from random offsets
                    for y in range(y_offset, h - self.patch_size + 1, self.stride):
                        for x in range(x_offset, w - self.patch_size + 1, self.stride):
                            patch_box = (x, y, x + self.patch_size, y + self.patch_size)
                            self.pairs.append((orig_path, proc_path, patch_box))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.use_sliding_window:
            orig_path, proc_path, patch_box = self.pairs[idx]
            orig_img = Image.open(orig_path).convert('L')
            proc_img = Image.open(proc_path).convert('L')
            if orig_img.size != proc_img.size:
                raise ValueError(f"Different sizes: {orig_path} vs {proc_path}")
            
            orig_patch = orig_img.crop(patch_box)
            proc_patch = proc_img.crop(patch_box)
        else:
            orig_path, proc_path = self.pairs[idx]
            orig_img = Image.open(orig_path).convert('L')
            proc_img = Image.open(proc_path).convert('L')

            if orig_img.size[0] < self.patch_size or orig_img.size[1] < self.patch_size:
                raise ValueError(f"Image too small: {orig_path} ({orig_img.size})")
            if orig_img.size != proc_img.size:
                raise ValueError(f"Different sizes: {orig_path} vs {proc_path}")

            # Seleção de patches híbrida (70% edges, 30% aleatório)
            patch_box = self._select_patch(np.array(orig_img))
            orig_patch = orig_img.crop(patch_box)
            proc_patch = proc_img.crop(patch_box)
        
        if self.transform:
            orig_patch, proc_patch = self.transform(orig_patch, proc_patch)

        return orig_patch, proc_patch

    def _select_patch(self, img_np):
        h, w = img_np.shape[:2]  
        if random.random() < self.edge_prob:
            edges = cv2.Canny(img_np, 50, 150)
            for _ in range(50):
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size) 
                edge_patch = edges[y:y+self.patch_size, x:x+self.patch_size]
                if edge_patch.mean() > 3:
                    return (x, y, x+self.patch_size, y+self.patch_size)
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size) 
        return (x, y, x+self.patch_size, y+self.patch_size)


if __name__ == "__main__":
    # Exemplo de uso do modo híbrido (seleção aleatória com foco em bordas)
    train_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/train/original",
        processed_dir="../data/frames_y/train/qp63",
        patch_size=64,
        patches_per_frame=6,
        transform=transforms.ToTensor(),
        use_sliding_window=False  # Hybrid mode
    )
    print("Number of training pairs (hybrid mode):", len(train_dataset))

    # Exemplo de uso do modo sliding window (todos os patches com stride definido)
    train_dataset_sw = OnlinePatchDataset(
        original_dir="../data/frames_y/train/original",
        processed_dir="../data/frames_y/train/qp63",
        patch_size=64,
        transform=transforms.ToTensor(),
        use_sliding_window=True,
        stride=32  # Can adjust stride as needed
    )
    print("Number of training pairs (sliding window):", len(train_dataset_sw))

    # Exemplo de uso para o dataset de validação (mantendo o modo híbrido)
    val_dataset = OnlinePatchDataset(
        original_dir="../data/frames_y/val/original",
        processed_dir="../data/frames_y/val/qp63",
        patch_size=240,
        patches_per_frame=2,
        transform=transforms.ToTensor(),
        use_sliding_window=False
    )
    print("Number of validation pairs:", len(val_dataset))

class PairedTransform:
    def __init__(self, augment=True, patch_size=64):
        self.augment = augment
        self.patch_size = patch_size
        self.angle_range = (-15, 15)
        self.hflip_p = 0.5
        self.vflip_p = 0.5
        self.crop_scale = (0.8, 1.0)
        
    def __call__(self, orig_patch, proc_patch):
        # Converter para PIL Image se necessário
        if not isinstance(orig_patch, Image.Image):
            orig_patch = transforms.ToPILImage()(orig_patch)
        if not isinstance(proc_patch, Image.Image):
            proc_patch = transforms.ToPILImage()(proc_patch)

        if self.augment:
            # Horizontal flip
            if random.random() < self.hflip_p:
                orig_patch = transforms.functional.hflip(orig_patch)
                proc_patch = transforms.functional.hflip(proc_patch)
            
            # Vertical flip
            if random.random() < self.vflip_p:
                orig_patch = transforms.functional.vflip(orig_patch)
                proc_patch = transforms.functional.vflip(proc_patch)
            
            # Rotação aleatória
            angle = random.uniform(*self.angle_range)
            orig_patch = transforms.functional.rotate(orig_patch, angle)
            proc_patch = transforms.functional.rotate(proc_patch, angle)
            
            # Random crop
            scale = random.uniform(*self.crop_scale)
            new_size = int(self.patch_size * scale)
            i = random.randint(0, self.patch_size - new_size)
            j = random.randint(0, self.patch_size - new_size)
            
            orig_patch = transforms.functional.crop(orig_patch, i, j, new_size, new_size)
            proc_patch = transforms.functional.crop(proc_patch, i, j, new_size, new_size)
            
            orig_patch = transforms.functional.resize(orig_patch, (self.patch_size, self.patch_size))
            proc_patch = transforms.functional.resize(proc_patch, (self.patch_size, self.patch_size))

        # Converter para tensor
        orig_tensor = transforms.ToTensor()(orig_patch)
        proc_tensor = transforms.ToTensor()(proc_patch)
        
        return orig_tensor, proc_tensor