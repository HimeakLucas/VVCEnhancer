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

def update_params_file(params_file, base_params, best_epoch, best_improvement):
    
    with open(params_file, "w") as f:
        f.write(base_params)
        f.write("\n")
        if best_epoch > 0:
            f.write(f"Best epoch so far: {best_epoch}\n")
            f.write(f"Best improvement (PSNR) so far: {best_improvement:.2f}%\n")
        else:
            f.write("No best epoch available yet.\n")

def train_model(model, optimizer, criterion, train_loader, val_loader, device, num_epochs, checkpoint_dir, params_file, base_params):
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

        # Treinamento
        model.train()
        running_loss = 0.0
        
        # Loop de treinamento com tqdm
        train_loop = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
        for (orig, proc) in train_loop:
            orig = orig.to(device)
            proc = proc.to(device)

            optimizer.zero_grad()
            output = model(proc)
            loss = criterion(output, orig)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Atualiza a barra de progresso com a perda atual
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        total_psnr = 0.0
        baseline_total_psnr = 0.0
        total_ssim = 0.0
        baseline_total_ssim = 0.0   
        val_loss = 0.0

        # Loop de validação com tqdm
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)
            for orig, proc in val_loop:
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
            f"\n[Epoch {epoch}/{num_epochs}] "
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

        if avg_baseline_ssim > 0:
            improvement = (avg_ssim - avg_baseline_ssim) / avg_baseline_ssim * 100
        else:
            improvement = 0.0
        print(f"  >> Improvement SSIM: {improvement:.2f}%")

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

        # Atualiza o arquivo de parâmetros com as informações da melhor época até o momento
        update_params_file(params_file, base_params, best_epoch, best_improvement)

    print(f"\nBest model was from epoch {best_epoch} with improvement {best_improvement:.2f}%")
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_psnrs": val_psnrs,
        "val_ssims": val_ssims,
        "baseline_psnrs": baseline_psnrs,
        "baseline_ssims": baseline_ssims,
        "best_epoch": best_epoch,
        "best_improvement": best_improvement
    }

def main():
    # Parâmetros de treinamento
    seed = 42
    set_seed(seed)

    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    grad_weight = 0.2
    patchs_per_frame = 5 # considerando sliding window desativado para validação
    patch_size = 240
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stride = 240  # considerando sliding window ativado para treinamento
    edge_prob = 0

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
    print("Training dataset: ")
    train_dataset = OnlinePatchDataset(
        original_dir=train_original_dir,
        processed_dir=train_processed_dir,
        patch_size=patch_size,
        patches_per_frame=patchs_per_frame,
        transform=transform,
        use_sliding_window=True,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # Cria uma pasta para o experimento usando data e hora
    experiment_folder = os.path.join("../checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(experiment_folder, exist_ok=True)
    print(f"Experiment: {experiment_folder}")

    # Salva os parâmetros do experimento em um arquivo de texto
    params_file = os.path.join(experiment_folder, "params.txt")
    base_params = f"""seed: {seed}
num_epochs: {num_epochs}
batch_size: {batch_size}
learning_rate: {learning_rate}
grad_weight: {grad_weight}
patchs_per_frame: {patchs_per_frame}
patch_size: {patch_size}
device: {device}
stride: {stride}
edge_prob: {edge_prob}
train_original_dir: {train_original_dir}
train_processed_dir: {train_processed_dir}
val_original_dir: {val_original_dir}
val_processed_dir: {val_processed_dir}
"""

    # Treinamento
    results = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=experiment_folder,
        params_file=params_file,
        base_params=base_params
    )

if __name__ == "__main__":
    main()
