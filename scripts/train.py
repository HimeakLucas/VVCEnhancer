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

# Adiciona o diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from models.ar_cnn import AR_CNN, QuickLoss  
from utils.online_patch_dataset import OnlinePatchDataset
from utils.metrics import calculate_psnr, calculate_ssim

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, optimizer, criterion, train_loader, val_loader, device, num_epochs, checkpoint_dir):
    # Listas para armazenar métricas de cada época
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    baseline_psnrs = []
    baseline_ssims = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_improvement = -float('inf')
    best_epoch = -1

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
                print(f"  [Train] Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")

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

        # Calcula a melhoria percentual de PSNR (se o baseline for maior que zero)
        if avg_baseline_psnr > 0:
            improvement = (avg_psnr - avg_baseline_psnr) / avg_baseline_psnr * 100
        else:
            improvement = 0.0
        print(f"  >> Improvement PSNR: {improvement:.2f}%")

        # Salva o melhor modelo se houve melhora na métrica de melhoria percentual de PSNR
        if improvement > best_improvement:
            best_improvement = improvement
            best_epoch = epoch
            best_model_path = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Best model updated at epoch {epoch} with improvement {best_improvement:.2f}%, saved at {best_model_path}")

        # Salva (ou atualiza) o modelo da última época a cada época
        last_model_path = os.path.join(checkpoint_dir, "model_last.pth")
        torch.save(model.state_dict(), last_model_path)
        print(f"  >> Last model updated at epoch {epoch}, saved at {last_model_path}")

    print(f"\nBest model was from epoch {best_epoch} with improvement {best_improvement:.2f}%")
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
    batch_size = 128
    learning_rate = 1e-3
    grad_weight = 0.5
    patchs_per_frame = 5  # considerando sliding window desativado para validação
    patch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stride = 128  # considerando sliding window ativado para treinamento

    model = AR_CNN().to(device)
    criterion = QuickLoss(grad_weight=grad_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    transform = transforms.ToTensor()

    # Diretórios dos dados
    train_original_dir = "../data/frames_y/train/original"
    train_processed_dir = "../data/frames_y/train/qp47"
    val_original_dir = "../data/frames_y/val/original"
    val_processed_dir = "../data/frames_y/val/qp47"

    # Criação dos datasets
    train_dataset = OnlinePatchDataset(
        original_dir=train_original_dir,
        processed_dir=train_processed_dir,
        patch_size=patch_size,
        patches_per_frame=patchs_per_frame,
        transform=transform,
        use_sliding_window=True,
        stride=stride
    )
    print("Training pairs (sliding window):", len(train_dataset))

    val_dataset = OnlinePatchDataset(
        original_dir=val_original_dir,
        processed_dir=val_processed_dir,
        patch_size=patch_size,
        patches_per_frame=patchs_per_frame,
        transform=transform,
        use_sliding_window=False,
    )
    print("Validation pairs:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # Cria uma pasta para o experimento usando data e hora
    experiment_folder = os.path.join("../checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"Experiment: {experiment_folder}")

    # Salva os parâmetros do experimento em um arquivo de texto
    params = f"""seed: {seed}
num_epochs: {num_epochs}
batch_size: {batch_size}
learning_rate: {learning_rate}
grad_weight: {grad_weight}
patchs_per_frame: {patchs_per_frame}
patch_size: {patch_size}
device: {device}
stride: {stride}
train_original_dir: {train_original_dir}
train_processed_dir: {train_processed_dir}
val_original_dir: {val_original_dir}
val_processed_dir: {val_processed_dir}
"""
    params_file = os.path.join(experiment_folder, "params.txt")
    with open(params_file, "w") as f:
        f.write(params)
    print(f"Parameters saved at: {params_file}")

    # Treinamento
    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=experiment_folder
    )

if __name__ == "__main__":
    main()
