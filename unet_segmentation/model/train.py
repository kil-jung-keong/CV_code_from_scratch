import matplotlib.pyplot as plt
import torch 
from torchvision.datasets import SBDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image 
import numpy as np
import os

class Transform:
    def __init__(self, resize=(256,256)):
        self.resize = resize
        self.img_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    def __call__(self, image, target):
        image = transforms.functional.resize(image, self.resize, interpolation=Image.BILINEAR)
        target = transforms.functional.resize(target, self.resize, interpolation=Image.NEAREST)

        image = self.img_tf(image)
        target = torch.from_numpy(np.array(target)).long()
        return image, target

def save_segmentation_batch(masks_tensor, output_dir='masks', cmap_name='tab20', prefix='mask'):
    """
    Segmentation 마스크 배치를 이미지 파일로 저장하는 함수

    Args:
        masks_tensor (Tensor): (B, H, W) 형태의 LongTensor
        output_dir (str): 저장할 디렉토리 이름
        cmap_name (str): 컬러맵 이름 (e.g., 'tab20', 'nipy_spectral', 'gist_ncar')
        prefix (str): 저장되는 파일명 접두어
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(masks_tensor.size(0)):
        mask_np = masks_tensor[i].detach().cpu().numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(mask_np, cmap=cmap_name)
        plt.axis('off')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, f'{prefix}_{i}.png')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"✅ 저장 완료: {file_path}")

train_dataset = SBDataset(
    root='../data',
    image_set = 'train',
    mode='segmentation',
    download=False,
    transforms=Transform(resize=(256,256))
)

train_img_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
train_mask_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

val_dataset = SBDataset(
    root='../data',
    image_set = 'val',
    mode='segmentation',
    download=False,
    transforms=Transform(resize=(256,256))
)

val_img_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
val_mask_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

for images, labels in train_img_loader:
    print(images.shape)
    save_segmentation_batch(labels,output_dir='saved_masks')
    break