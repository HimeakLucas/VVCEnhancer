import torch
import torch.nn as nn

class ForwardBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(ForwardBlock, self).__init__()
        # Convolução 1x1 para projeção
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.PReLU(num_parameters=out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out

class FeatureExtractionBlock(nn.Module):
    def __init__(self, channels=128):
        super(FeatureExtractionBlock, self).__init__()
        # Convolução 3x3 com padding 1 para manter as dimensões espaciais
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.PReLU(num_parameters=channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out

class TailBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(TailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Tanh()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        return out

class PostProcessingNet(nn.Module):
    def __init__(self, in_channels=1, num_feat_blocks=16):

        super(PostProcessingNet, self).__init__()

        self.forward_block = ForwardBlock(in_channels=in_channels, out_channels=128)

        self.feat_blocks = nn.Sequential(*[FeatureExtractionBlock(128) for _ in range(num_feat_blocks)])

        self.tail_block = TailBlock(in_channels=128, out_channels=in_channels)
    
    def forward(self, x):

        residual = x

        out = self.forward_block(x)

        out = self.feat_blocks(out)

        out = self.tail_block(out)

        out = residual + out
        
        return out