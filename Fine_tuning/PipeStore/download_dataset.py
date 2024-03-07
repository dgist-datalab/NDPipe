import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import numpy as np
import os

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000)
testloader  = torch.utils.data.DataLoader(testset, batch_size=1000)

with open('./dataset/train_image.dat', 'ab') as f_image, open('./dataset/train_label.dat', 'ab') as f_label:
    for batch in tqdm(trainloader, desc = 'train data'):
        f_image.write(batch[0].numpy().astype(np.float16).tobytes())
        f_label.write(batch[1].numpy().astype(np.uint32).tobytes())

with open('./dataset/test_image.dat', 'ab') as f_image, open('./dataset/test_label.dat', 'ab') as f_label:
    for batch in tqdm(testloader, desc = 'test data'):
        f_image.write(batch[0].numpy().astype(np.float16).tobytes())
        f_label.write(batch[1].numpy().astype(np.uint32).tobytes())

os.system('rm -rf ./dataset/cifar-100-python')
os.system('rm -rf ./dataset/cifar-100-python.tar.gz')
