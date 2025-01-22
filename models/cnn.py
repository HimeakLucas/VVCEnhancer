from utils.metrics import calculate_psnr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np

tile_size = (64, 64) 
#batch_size = 16
#learning_rate = 0.001
#num_epochs = 10

class VideoDataset(Dataset):
    def __init__(self, raw_frames_path, compressed_frames_path, tile_size):
        self.raw_path = raw_frames_path
        self.compressed_path = compressed_frames_path
        self.tile_size = tile_size

        self.raw_files = sorted(os.listdir(raw_frames_path))
        self.compressed_files = sorted(os.listdir(compressed_frames_path))

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):

        raw_frame = cv2.imread(os.path.join(self.raw_path, self.raw_files[idx]), cv2.IMREAD_GRAYSCALE)
        compressed_frame = cv2.imread(os.path.join(self.compressed_path, self.compressed_files[idx]), cv2.IMREAD_GRAYSCALE)

        # Standardize size and split into tiles
        raw_frame = self.add_padding(raw_frame)
        compressed_frame = self.add_padding(compressed_frame)

        raw_tiles = self.split_tiles(raw_frame)
        compressed_tiles = self.split_tiles(compressed_frame)

        # Convert to tensors
        raw_tiles = torch.tensor(raw_tiles, dtype=torch.float32).unsqueeze(1) / 255.0
        compressed_tiles = torch.tensor(compressed_tiles, dtype=torch.float32).unsqueeze(1) / 255.0

        return compressed_tiles, raw_tiles

    def add_padding(self, frame):
        h, w = frame.shape
        pad_h = (self.tile_size[0] - (h % self.tile_size[0])) % self.tile_size[0]
        pad_w = (self.tile_size[1] - (w % self.tile_size[1])) % self.tile_size[1]
        return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    def split_tiles(self, frame):
        h, w = frame.shape
        tiles = [
            frame[i:i+self.tile_size[0], j:j+self.tile_size[1]]
            for i in range(0, h, self.tile_size[0])
            for j in range(0, w, self.tile_size[1])
        ]
        return np.array(tiles)
    
class ArtifactReductionNet(nn.Module):
    def __init__(self):
        super(ArtifactReductionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArtifactReductionNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_loader = DataLoader(VideoDataset('../data/raw_frames/', '../compressed_frames/', (64, 64)), batch_size=16, shuffle=True)

for epoch in range(10):
    for compressed_tiles, raw_tiles in train_loader:
        compressed_tiles, raw_tiles = compressed_tiles.to(device), raw_tiles.to(device)

        optimizer.zero_grad()
        outputs = model(compressed_tiles)
        loss = criterion(outputs, raw_tiles)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")