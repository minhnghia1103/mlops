import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

class Cutout:
    def __init__(self, size):
        self.size = size  # Kích thước ô vuông che đi

    def __call__(self, img):
        # Convert to numpy
        img_np = np.array(img)

        # Randomly select a position
        h, w, _ = img_np.shape
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Define the square region
        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        # Apply cutout
        img_np[y1:y2, x1:x2, :] = 0
        return Image.fromarray(img_np)


transform_train = Compose([
    RandomHorizontalFlip(),  # Lật ngẫu nhiên
    ToTensor(),
    Cutout(size=16),  # Áp dụng cutout với kích thước ô vuông 16x16
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa
])

transform_test = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Tạo dataset và DataLoader
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Lưu dữ liệu đã augment vào file .pt
torch.save(trainset, './data/trainset.pt')
torch.save(testset, './data/testset.pt')