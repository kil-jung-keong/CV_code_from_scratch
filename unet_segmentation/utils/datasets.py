import torch 
from torchvision.datasets import SBDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [lambda x: torch.from_numpy(x).long(),
    transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

dataset = SBDataset(
    root='../data',
    image_set = 'train',
    mode='boundaries',
    download=True,
    transforms=transform,
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)