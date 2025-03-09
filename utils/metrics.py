import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(output, target, max_value=1.0):
    #Assume que os valores estão normalizados entre 0 e max_value.
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    psnr_val = 10 * torch.log10((max_value ** 2) / mse)
    return psnr_val.item()

def calculate_ssim(output, target, max_value=1.0):
    # Assume que as imagens são tensores com shape [B, 1, H, W] e valores em [0, max_value].
    # B (Batch Size): Número de imagens processadas simultaneamente.
    # 1 (Número de Canais): 1 = Escala de cinza, 3 = RGB
    # H (Height)
    # W (Width)
    
    ssim_total = 0.0
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    batch = output_np.shape[0]
    for i in range(batch):
        # Como as imagens são em escala de cinza, usamos o primeiro canal.
        out_img = output_np[i, 0, :, :]
        tar_img = target_np[i, 0, :, :]
        # Calcular SSIM; a função espera que os valores estejam na escala [0, max_value]
        ssim_value = ssim(tar_img, out_img, data_range=max_value)
        ssim_total += ssim_value
    return ssim_total / batch
