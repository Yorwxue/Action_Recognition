import os
import torch
from torch.utils.data import DataLoader

from Dataset import UCF101
from Model import CreateModel

model_path = os.path.abspath("./checkpoint/InceptionV3_UCF.pt")

ucf101_dataset = UCF101(sample_num=1)

use_cuda = True
if use_cuda:
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

model = CreateModel(device, in_channels=3, num_labels=101)

dataloader = DataLoader(ucf101_dataset, batch_size=8, shuffle=True, num_workers=5)
model.train(dataloader, device, num_epoch=15, display_freq=500, model_path=model_path)

# test
print("---------- test ------------")
ucf101_dataset.training(False)

dataloader = DataLoader(ucf101_dataset, batch_size=8, shuffle=False, num_workers=5)

model.load_model(CreateModel, model_path=model_path)
model.test(dataloader, device)

pass
