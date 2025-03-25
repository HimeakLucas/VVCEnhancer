import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import sys
import os
from datetime import datetime
import random
from tqdm import tqdm
import mlflow
import mlflow.pytorch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from models.ar_cnn import AR_CNN, QuickLoss  
from utils.online_patch_dataset import OnlinePatchDataset
from utils.metrics import calculate_psnr, calculate_ssim

from torchinfo import summary

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_loader, optmizer, criterion, device):
    model.train()
    total_loss = 0

    train_loop = tqdm(train_loader, desc=f"Training", leave=False)
    for batch, (original, processed) in enumerate(train_loop):
        original = original.to(device)
        processed = processed.to(device)

        optmizer.zero_grad()
        output = model(processed)
        loss = criterion(output, original)
        loss.backward()
        optmizer.step()

        total_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    
    total_psnr = 0
    total_ssim = 0

    total_baseline_psnr = 0
    total_baseline_ssim = 0

    total_loss = 0
    val_loop = tqdm(val_loader, desc=f"Validation", leave=False)
    
    with torch.no_grad():
        for batch, (original, processed) in enumerate(val_loop):
            original = original.to(device)
            processed = processed.to(device)

            output = model(processed)
            loss = criterion(output, original)

            total_psnr += calculate_psnr(output, original)
            total_ssim += calculate_ssim(output, original)

            total_baseline_psnr += calculate_psnr(processed, original)
            total_baseline_ssim += calculate_ssim(processed, original)

            val_loop.set_postfix(loss=loss.item())

    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    avg_baseline_psnr = total_baseline_psnr / len(val_loader)
    avg_baseline_ssim = total_baseline_ssim / len(val_loader)

    avg_loss = total_loss / len(val_loader)

    return avg_psnr, avg_ssim, avg_baseline_psnr, avg_baseline_ssim, avg_loss

def main():

    seed = 42
    set_seed(seed)

    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    grad_weight = 0
    patchs_per_frame = 2  # considerando sliding window desativado para validação
    patch_size = 240
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stride = 240  # considerando sliding window ativado para treinamento
    edge_prob = 0

    model = AR_CNN().to(device)
    criterion = QuickLoss(grad_weight=grad_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    transform = transforms.ToTensor()

    train_original_dir = "../data/frames_y/train/original"
    train_processed_dir = "../data/frames_y/train/qp47"
    val_original_dir = "../data/frames_y/val/original"
    val_processed_dir = "../data/frames_y/val/qp47"

    print("Training dataset: ")
    train_dataset = OnlinePatchDataset(
        original_dir=train_original_dir,
        processed_dir=train_processed_dir,
        patch_size=patch_size,
        patches_per_frame=patchs_per_frame,
        transform=transform,
        use_sliding_window=False,
        edge_prob=edge_prob,
        stride=stride
    )

    print("Validation dataset: ")
    val_dataset = OnlinePatchDataset(
        original_dir=val_original_dir,
        processed_dir=val_processed_dir,
        patch_size=patch_size,
        patches_per_frame=2,
        transform=transform,
        use_sliding_window=False,
        edge_prob=edge_prob
    )

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    with mlflow.start_run():
        params = {
            "seed": seed,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "grad_weight": grad_weight,
            "patchs_per_frame": patchs_per_frame,
            "patch_size": patch_size,
            "stride": stride,
            "edge_prob": edge_prob,
            "loss": criterion.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
        }
        mlflow.log_params(params)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for epoch in range(num_epochs):
            start_time = time.time()
            print(f"\n[Epoch {epoch}/{num_epochs}] ")
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_psnr, val_ssim, val_baseline_psnr, val_baseline_ssim, val_loss = validate(model, val_loader, criterion, device)
            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"  >> Train Loss: {train_loss:.4f}")
            print(f"  >> Val   Loss: {val_loss:.4f}")
            print(f"  >> Val   PSNR (Rede): {val_psnr:.2f} dB | Baseline: {val_baseline_psnr:.2f} dB")
            print(f"  >> Val   SSIM (Rede): {val_ssim:.4f}   | Baseline: {val_baseline_ssim:.4f}")
            print(f"  >> Epoch duration: {epoch_duration:.2f} s")

            psnr_improvement = 0
            ssim_improvement = 0

            if val_baseline_psnr > 0:
                psnr_improvement = (val_psnr - val_baseline_psnr) / val_baseline_psnr * 100
            else:
                psnr_improvement = 0.0
            print(f"  >> Improvement PSNR: {psnr_improvement:.2f}%")

            if val_baseline_ssim > 0:
                ssim_improvement = (val_ssim - val_baseline_ssim) / val_baseline_ssim * 100
            else:
                ssim_improvement = 0.0
            print(f"  >> Improvement SSIM: {ssim_improvement:.2f}%")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_baseline_psnr": val_baseline_psnr,
                "val_baseline_ssim": val_baseline_ssim,
                "psnr_improvement": psnr_improvement,
                "ssim_improvement": ssim_improvement
            }, step=epoch)

            mlflow.pytorch.log_model(model, "models")

if __name__ == "__main__":
    main()
