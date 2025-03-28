import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """Defines a single block of the U-Net Generator with optional dropout"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            if use_dropout:
                self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, x):
        return self.conv(x)

class GeneratorUNet(nn.Module):
    """U-Net-based Generator"""
    def __init__(self, input_channels=1, output_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = UNetBlock(input_channels, 64, down=True, use_dropout=False)
        self.enc2 = UNetBlock(64, 128, down=True, use_dropout=False)
        self.enc3 = UNetBlock(128, 256, down=True, use_dropout=False)
        self.enc4 = UNetBlock(256, 512, down=True, use_dropout=True)
        self.enc5 = UNetBlock(512, 512, down=True, use_dropout=True)
        self.enc6 = UNetBlock(512, 512, down=True, use_dropout=True)
        self.enc7 = UNetBlock(512, 512, down=True, use_dropout=False)
        self.enc8 = UNetBlock(512, 512, down=True, use_dropout=False)

        # Decoder (upsampling)
        self.dec1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.dec2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.dec4 = UNetBlock(1024, 512, down=False, use_dropout=False)
        self.dec5 = UNetBlock(1024, 256, down=False, use_dropout=False)
        self.dec6 = UNetBlock(512, 128, down=False, use_dropout=False)
        self.dec7 = UNetBlock(256, 64, down=False, use_dropout=False)
        self.dec8 = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        d8 = self.dec8(torch.cat([d7, e1], dim=1))
        
        return d8
class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, input_channels=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, grayscale, colorized):
        x = torch.cat([grayscale, colorized], dim=1)
        return self.model(x)
