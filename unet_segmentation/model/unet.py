import torch 
import os 
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 

# GPU 장치 선언 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# skip connection을 위한 center crop 
def center_crop(enc_feat, target_size):
    _, _, h, w = enc_feat.shape 
    th, tw = target_size 
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return enc_feat[:, :, y1:y1+th, x1:x1+tw]

# convolution block 정의하기 - 3x3 conv + relu 2번
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
        
class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.down_conv1 = ConvolutionBlock(3, 64)
        self.down_conv2 = ConvolutionBlock(64, 128)
        self.down_conv3 = ConvolutionBlock(128, 256)
        self.down_conv4 = ConvolutionBlock(256, 512)
        self.down_conv5 = ConvolutionBlock(512, 1024)
        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvolutionBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvolutionBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvolutionBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvolutionBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down_conv1(x)
        print("x1:",x1.shape)  # [1, 64, 568, 568]
        x2 = self.down_conv2(self.pool(x1))
        print("x2:",x2.shape) # [1, 128, 280, 280]
        x3 = self.down_conv3(self.pool(x2))
        print("x3:",x3.shape) # [1, 256, 136, 136]
        x4 = self.down_conv4(self.pool(x3))
        print("x4:",x4.shape) # [1, 512, 64, 64]
        out = self.pool(x4)
        print("out:",out.shape) # [1, 512, 32, 32]
        out = self.down_conv5(out)
        print("out:",out.shape) # [1, 1024, 28, 28]
        out = self.up_transpose1(out)
        print("out:",out.shape) # [1, 512, 56, 56]
        out = torch.cat((out, center_crop(x4, out.shape[2:])),dim=1)
        print("out:",out.shape) # [1, 1024, 56, 56]
        out = self.up_conv1(out)
        print("out:",out.shape) # [1, 512, 52, 52]
        out = self.up_transpose2(out)
        print("out:",out.shape) # [1, 256, 104, 104]
        out = torch.cat((out, center_crop(x3, out.shape[2:])),dim=1)
        print("out:",out.shape) # [1, 512, 104, 104]
        out = self.up_conv2(out)
        print("out:",out.shape) # [1, 256, 100, 100]
        out = self.up_transpose3(out)
        print("out:",out.shape) # [1, 128, 200, 200]
        out = torch.cat((out, center_crop(x2, out.shape[2:])),dim=1)
        print("out:",out.shape) # [1, 256, 200, 200]
        out = self.up_conv3(out)
        print("out:",out.shape) # [1, 128, 196, 196]
        out = self.up_transpose4(out)
        print("out:",out.shape) # [1, 64, 392, 392] 
        out = torch.cat((out, center_crop(x1, out.shape[2:])),dim=1)
        out = self.up_conv4(out)
        print("out:",out.shape) # [1, 64, 388, 388] 
        out = self.final(out)
        print("out:",out.shape) # [1, 2, 388, 388]
        return out 

model = UNet(num_classes=1).to(device)
x = torch.randn(1, 3, 572, 572).to(device)
outputs = model(x)
        



