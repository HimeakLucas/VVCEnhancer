import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from models.ar_cnn import AR_CNN, QuickLoss  
from utils.online_patch_dataset import OnlinePatchDataset
from utils.metrics import calculate_psnr, calculate_ssim

import os
import random
import numpy as np
import torch

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, 
                optimizer, 
                criterion, 
                train_loader, 
                val_loader, 
                device,
                num_epochs,
                checkpoint_dir="."):


    # Listas para armazenar métricas de cada época
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    baseline_psnrs = []
    baseline_ssims = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        print(f"\nSTARTING EPOCH {epoch}/{num_epochs}")

        # Treinamento
        model.train()
        running_loss = 0.0
        
        for batch_idx, (orig, proc) in enumerate(train_loader):
            orig = orig.to(device)
            proc = proc.to(device)

            optimizer.zero_grad()
            output = model(proc)
            loss = criterion(output, orig)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"  [Train] Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)} "
                      f"- Loss: {loss.item():.6f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        total_psnr = 0.0
        baseline_total_psnr = 0.0
        total_ssim = 0.0
        baseline_total_ssim = 0.0   
        val_loss = 0.0

        with torch.no_grad():
            for orig, proc in val_loader:
                orig = orig.to(device)
                proc = proc.to(device)
                
                output = model(proc)
                loss = criterion(output, orig)
                val_loss += loss.item()

                total_psnr += calculate_psnr(output, orig)
                total_ssim += calculate_ssim(output, orig)
                baseline_total_psnr += calculate_psnr(proc, orig)
                baseline_total_ssim += calculate_ssim(proc, orig)

        avg_val_loss = val_loss / len(val_loader)  
        val_losses.append(avg_val_loss)

        avg_psnr = total_psnr / len(val_loader)
        avg_baseline_psnr = baseline_total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_baseline_ssim = baseline_total_ssim / len(val_loader)

        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)
        baseline_psnrs.append(avg_baseline_psnr)
        baseline_ssims.append(avg_baseline_ssim)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        summary_msg = (
            f"\n[Época {epoch}/{num_epochs}] "
            f"\n  >> Train Loss: {avg_train_loss:.4f}"
            f"\n  >> Val   Loss: {avg_val_loss:.4f}"
            f"\n  >> Val   PSNR (Rede): {avg_psnr:.2f} dB | Baseline: {avg_baseline_psnr:.2f} dB"
            f"\n  >> Val   SSIM (Rede): {avg_ssim:.4f}   | Baseline: {avg_baseline_ssim:.4f}"
            f"\n  >> Epoch duration: {epoch_duration:.2f} s"
        )
        print(summary_msg)

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  >> Model saved at {checkpoint_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_psnrs": val_psnrs,
        "val_ssims": val_ssims,
        "baseline_psnrs": baseline_psnrs,
        "baseline_ssims": baseline_ssims
    }

def main():
    # Parâmetros de treinamento
    seed = 42
    set_seed(seed)

    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    grad_weight = 0
    patchs_per_frame = 20
    patch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AR_CNN().to(device)
    criterion = QuickLoss(grad_weight=grad_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    transform = transforms.ToTensor()

    # Diretórios dos dados
    train_original_dir = "../data/frames_y/train/original"
    train_processed_dir = "../data/frames_y/train/qp63"
    val_original_dir = "../data/frames_y/val/original"
    val_processed_dir = "../data/frames_y/val/qp63"

    # Criação dos datasets
    train_dataset = OnlinePatchDataset(
        original_dir = train_original_dir,
        processed_dir = train_processed_dir,
        patch_size = patch_size,
        patches_per_frame = patchs_per_frame,
        transform = transform
    )

    val_dataset = OnlinePatchDataset(
        original_dir = val_original_dir,
        processed_dir = val_processed_dir,
        patch_size = patch_size,
        patches_per_frame = patchs_per_frame,
        transform = transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= False, num_workers=4)

    # Treinamento
    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir="../checkpoints"
    )

if __name__ == "__main__":
    main()