import torch
import torch.nn as nn

class AR_CNN(nn.Module):
    def __init__(self):
        super(AR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, 7, padding=3)
        self.dropout = nn.Dropout2d(0.3)
        self.conv3 = nn.Conv2d(32, 1, 5, padding=2)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = out + residual
        out = torch.clamp(out, 0, 1)  # garante saída no intervalo [0,1]
        return out

class QuickLoss(nn.Module):
    def __init__(self, grad_weight=0.1):
        super(QuickLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.grad_weight = grad_weight
        
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        
        # Gradiente na direção x
        grad_x_out = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        grad_x_tar = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Gradiente na direção y
        grad_y_out = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        grad_y_tar = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        grad_loss_x = torch.mean(torch.abs(grad_x_out - grad_x_tar))
        grad_loss_y = torch.mean(torch.abs(grad_y_out - grad_y_tar))
        
        grad_loss = grad_loss_x + grad_loss_y
        
        return mse_loss + self.grad_weight * grad_loss
