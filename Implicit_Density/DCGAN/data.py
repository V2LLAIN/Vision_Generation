import random
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

device = torch.device("cuda:0" if (torch.cuda.is_available())
                      
img_size=64
batch_size=128
input_latent=100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.ToPILImage(),
     transforms.Scale(size=(img_size, img_size), interpolation=Image.BICUBIC),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder(root='./data_faces',transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=3, normalize=True).cpu(),(1,2,0)))
