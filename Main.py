import os
import torch
from torch.utils.data import DataLoader

from Dataset import UCF101, kinetic
from Model import CreateModel

model_path = os.path.abspath("./checkpoint/InceptionV3_kinetics_Spatial.pt")

kinetic_dataset = kinetic(sample_num=1)

use_cuda = True
if use_cuda:
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")

model = CreateModel(device, in_channels=3, num_labels=101)

dataloader = DataLoader(kinetic_dataset, batch_size=128, shuffle=True, num_workers=30)
model.train(dataloader, device, num_epoch=100, display_freq=500, model_path=model_path)

# test
print("---------- test ------------")
kinetic_dataset.training(False)

dataloader = DataLoader(kinetic_dataset, batch_size=128, shuffle=False, num_workers=30)

model.load_model(CreateModel, model_path=model_path)
model.test(dataloader, device)

pass
