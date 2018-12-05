import torch
from torch.utils.data import DataLoader

from Dataset import UCF101
from Model import CreateModel

ucf101_dataset = UCF101()
dataloader = DataLoader(ucf101_dataset, batch_size=5, shuffle=True, num_workers=5)

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model = CreateModel(device, in_channels=10, num_labels=101)

model.train(dataloader, device, num_epoch=10)

pass
