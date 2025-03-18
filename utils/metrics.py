import torch
import torch.nn.functional as F

def calculate_psnr(output, target, max_value=1.0):
    """
    Calcula o PSNR de forma vetorizada para cada imagem no batch.
    Retorna a média do PSNR para o batch.
    """
    mse = ((output - target) ** 2).view(output.shape[0], -1).mean(dim=1)
    mse = torch.clamp(mse, min=1e-10)  # Evita divisão por zero
    psnr_val = 10 * torch.log10((max_value ** 2) / mse)
    return psnr_val.mean().item()

def gaussian_window(window_size, sigma):
    """
    Cria uma janela gaussiana 1D.
    """
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Cria uma janela 2D gaussiana para convolução, com shape [channel, 1, window_size, window_size].
    """
    _1D_window = gaussian_window(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # Produto externo para criar o kernel 2D
    window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
    return window.contiguous()

def calculate_ssim(output, target, max_value=1.0, window_size=11):
    """
    Calcula o SSIM de forma vetorizada usando convolução com janela gaussiana.
    """
    (_, channel, _, _) = output.size()
    window = create_window(window_size, channel).to(output.device)

    # Média local usando convolução
    mu_output = F.conv2d(output, window, padding=window_size//2, groups=channel)
    mu_target = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu_output_sq = mu_output.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_output_target = mu_output * mu_target

    # Desvio padrão local
    sigma_output_sq = F.conv2d(output * output, window, padding=window_size//2, groups=channel) - mu_output_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu_target_sq
    sigma_output_target = F.conv2d(output * target, window, padding=window_size//2, groups=channel) - mu_output_target

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    # Cálculo do mapa SSIM
    ssim_map = ((2 * mu_output_target + C1) * (2 * sigma_output_target + C2)) / \
               ((mu_output_sq + mu_target_sq + C1) * (sigma_output_sq + sigma_target_sq + C2))
    
    # Média do SSIM no batch
    return ssim_map.mean(dim=[1, 2, 3]).mean().item()
